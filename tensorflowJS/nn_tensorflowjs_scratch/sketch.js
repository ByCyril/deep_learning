let weights;

function setup() {
  weights = tf.variable(
    tf.tensor([[2 * random(1)], [2 * random(1)], [2 * random(1)]])
  );

  weights.print();

  let inputs = tf.tensor([[0, 0, 1], [1, 1, 1], [0, 1, 1]]);
  let outputs = tf.tensor([[0], [1], [0]]);

  train(inputs, outputs, 10000);
}

function makePrediction() {
  let x = document.getElementById("values").value.split(" ");
  var a = parseInt(x[0]);
  var b = parseInt(x[1]);
  var c = parseInt(x[2]);
  let prediction = predict(tf.tensor([a, b, c]));
  prediction.print();
}

function sigmoid(x) {
  var data = x.dataSync();
  let results = [];

  for (let i = 0; i < data.length; i++) {
    let a = 1 / (1 + Math.pow(Math.E, -data[i]));
    results.push([a]);
  }

  return tf.tensor(results);
}

function sigmoidPrime(x) {
  var data = x.dataSync();
  let results = [];

  for (let i = 0; i < data.length; i++) {
    let a = data[i] * (1 - data[i]);
    results.push([a]);
  }

  return tf.tensor(results);
}

function train(inputs, outputs, iteration) {
  inputs.print();
  outputs.print();

  for (let i = 0; i < iteration; i++) {
    var output = predict(inputs);
    var error = outputs.sub(output);
    var a = error.mul(sigmoidPrime(output));
    let inputT = tf.transpose(inputs);
    let adjustments = inputT.dot(a);

    weights = weights.add(adjustments);
  }
  console.log("New weights");
  weights.print();
  console.log("done");
}

function predict(inputs) {
  let dot = inputs.dot(weights);
  return sigmoid(dot);
}
