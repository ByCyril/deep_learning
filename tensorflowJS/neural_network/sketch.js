// create the model

const model = tf.sequential();
// create the first hidden layer with 4 nodes taking in 3 inputs
// Uses the sigmoid activation function
const hidden = tf.layers.dense({
  units: 3,
  inputShape: [3],
  activation: "tanh"
});

// add the hidden layer
model.add(hidden);

// create the output layer. In this case, we will only ask for 1 output layer
// uses the sigmoid activation function
const output = tf.layers.dense({
  units: 1,
  activation: "sigmoid"
});
// add the output layer
model.add(output);

// set the learning rate
const sgdOpt = tf.train.sgd(0.2);

// compile with the learning rate and the type of error calculations
model.compile({
  optimizer: sgdOpt,
  loss: tf.losses.meanSquaredError
});

// training inputs
const xs = tf.tensor2d([
  [0, 1, 0],
  [0, 1, 1],
  [1, 1, 0],
  [-1, 1, 0],
  [-1, 0, 0],
  [-1, 0, 1],
  [1, 0, 1],
  [0, 1, 0],
  [0, 0, 0],
  [-1, 0, 0],
  [0, 0, 1],
  [1, 1, 1],
  [1, 0, 0],
  [-1, 1, 1]
]);
// training outputs
const ys = tf.tensor2d([
  [0],
  [0],
  [1],
  [1],
  [1],
  [0],
  [1],
  [0],
  [1],
  [1],
  [1],
  [1],
  [1],
  [0]
]);

// make a prediction
train().then(function() {
  console.log("training complete");
  let testData = tf.tensor2d([-1, 1, 0]);
  let y = model.predict(xs);

  y.print();
});

// train the model

async function train() {
  model.layers[0].setWeights([
    [-2.2497578, -1.4616488, -0.2667132],
    [1.9590461, -0.806393, 0.5483124],
    [-0.1415037, -2.1341474, 0.4887933]
  ]);

  // for (let i = 0; i < 1000; i++) {
  //   const response = await model.fit(xs, ys);

  //   console.log(i, response.history.loss[0]);
  // }

  // console.log("Weights: ");
  // model.layers[0].getWeights()[0].print();
}
