// create the model

const model = tf.sequential();
// create the first hidden layer with 4 nodes taking in 3 inputs
// Uses the sigmoid activation function
const hidden = tf.layers.dense({
  units: 4,
  inputShape: [3],
  activation: "sigmoid"
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
const sgdOpt = tf.train.sgd(0.5);

// compile with the learning rate and the type of error calculations
model.compile({
  optimizer: sgdOpt,
  loss: tf.losses.meanSquaredError
});

// training inputs
const xs = tf.tensor2d([[0, 0, 1], [1, 1, 1], [0, 1, 1]]);
// training outputs
const ys = tf.tensor2d([[0], [1], [0]]);

// make a prediction
train().then(function() {
  console.log("training complete");
  let y = model.predict(xs);

  y.print();
});

// train the model
async function train() {
  for (let i = 0; i < 10000; i++) {
    const response = await model.fit(xs, ys);
    console.log(i, response.history.loss[0]);
  }
}
