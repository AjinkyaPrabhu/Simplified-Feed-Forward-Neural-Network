import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_classes = 10
batch_size = 100


x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


number_of_hidden_layers = 2 # Modify according to your wish



hidden_layers = []
class layers:
    def __init__(self,rows,cols):
        self.W = tf.Variable(tf.random_normal([rows,cols]))
        self.b = tf.Variable(tf.random_normal([cols]))
        self.rows = rows
        self.col = cols

def initialise_nn():

    #Creates the hidden Layers ** FIRST LAYER ROWS MUST EQUAL NUMBER OF INPUT FEATURES**
    for i in range(1,number_of_hidden_layers):
        hl_dims = [int(x) for x in input("Enter dimensions of hidden layer **Please ensure Cols"
                                    " of previous are equal to rows of this**\n").split(' ')]
        hidden_layers.append(layers(hl_dims[0],hl_dims[1]))
    print(hidden_layers)


def nn_model(x):
    acc = tf.add(tf.matmul(x,hidden_layers[0].W),hidden_layers[0].b)
    acc = tf.nn.relu(acc)
    output_layer = {"W":tf.Variable(tf.random_normal([hidden_layers[number_of_hidden_layers-1].col,n_classes])),
              "b":tf.Variable(tf.random_normal([n_classes]))}

    for i in range(1,number_of_hidden_layers):
        acc = tf.add(tf.matmul(acc, hidden_layers[i].W),
                     hidden_layers[i].b)
        acc = tf.nn.relu(acc)

    output = tf.add(tf.matmul(acc, output_layer["W"]),
                    output_layer["b"])

    return output





##Modify this function to train
def train_neural_network(x):
    prediction = nn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost) #Choose apprpriate optimizer

    hm_epochs = 10 # Number of Epochs
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

initialise_nn()
train_neural_network(x)

