'''
this is l2_attack using tensorflow2.0
'''
import sys
import tensorflow as tf
import numpy as np

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 10       # the initial constant c to pick as a first guess

# IMAGE_SIZE = 320         # wight and height of inputting model's images
# NUM_CHANNELS = 3         # images' channels
# NUM_LABELS = 20          # number of labels


class CarliniL2:
    def __init__(self, model, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST,
                 boxmin=-0.5, boxmax=0.5):
        """
        The L_2 optimized attack.

        This attack is the most efficient and should be used as the primary
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """
        self.boxmin = boxmin
        self.boxmax = boxmax
        self.model = model

        self.image_size, self.num_channels, self.num_labels = model.image_size, model.num_channels, model.num_labels
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size

        self.repeat = binary_search_steps >= 10

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False

        self.shape = (batch_size, self.image_size, self.image_size, self.num_channels)

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmul = (self.boxmax - self.boxmin) / 2.
        self.boxplus = (self.boxmin + self.boxmax) / 2.

        self.optimizer = tf.keras.optimizers.Adam(self.LEARNING_RATE)

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to', len(imgs))
        for i in range(0, len(imgs), self.batch_size):
            print('tick', i)
            r.extend(self.attack_batch(imgs[i:i + self.batch_size], targets[i:i + self.batch_size]))
        return np.array(r)

    def get_grad(self, imgs, labs, modifier, const):
        with tf.GradientTape() as tape:
            # the resulting image, tanh'd to keep bounded from boxmin to boxmax
            self.newimg = tf.tanh(modifier + imgs) * self.boxmul + self.boxplus

            # prediction BEFORE-SOFTMAX of the model
            self.output = self.model.predict_logits(self.newimg)

            # distance to the input data
            self.l2dist = tf.reduce_sum(tf.square(self.newimg - (tf.tanh(imgs) * self.boxmul + self.boxplus)),
                                        [1, 2, 3])

            # compute the probability of the label class versus the maximum other
            real = tf.reduce_sum((labs) * self.output, 1)
            other = tf.reduce_max((1 - labs) * self.output - (labs * 10000), 1)

            if self.TARGETED:
                # if targetted, optimize for making the other class most likely
                loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
            else:
                # if untargeted, optimize for making this class least likely.
                loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

            # sum up the losses
            self.loss2 = tf.reduce_sum(self.l2dist)
            self.loss1 = tf.reduce_sum(const * loss1)
            self.loss = self.loss1 + self.loss2

        grads = tape.gradient(self.loss, modifier)
        return grads, modifier

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # convert to tanh-space
        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(o_bestl2)
            # completely reset adam's internal state.
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]
            batch = tf.convert_to_tensor(batch, dtype=tf.float32)
            batchlab = tf.convert_to_tensor(batchlab, dtype=tf.float32)

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            prev = np.inf
            modifier = tf.Variable(initial_value=tf.zeros(self.shape, dtype=tf.float32), trainable=True)
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                grads, modifier = self.get_grad(batch, batchlab, modifier, CONST)
                self.optimizer.apply_gradients(zip([grads], [modifier]))

                l, l2s, scores, nimg = self.loss, self.l2dist, self.output, self.newimg

                if np.all(scores >= -.0001) and np.all(scores <= 1.0001):
                    if np.allclose(np.sum(scores, axis=1), 1.0, atol=1e-3):
                        if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                            raise Exception(
                                "The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")

                # print out the losses every 10%
                if iteration % (self.MAX_ITERATIONS // 10) == 0:
                    print(iteration, (self.loss.numpy(), self.loss1.numpy(), self.loss2.numpy()))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if l > prev * .9999:
                        break
                    prev = l

                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack
