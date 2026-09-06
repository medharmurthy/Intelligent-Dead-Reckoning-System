const ort = require('onnxruntime-node');

async function check(path) {
  try {
    const session = await ort.InferenceSession.create(path);
    console.log('================');
    console.log('Model:', path);
    console.log('Inputs:');
    session.inputNames.forEach(n => console.log(' ', n, 'Shape:', session.inputs[n]));
    console.log('Outputs:');
    session.outputNames.forEach(n => console.log(' ', n, 'Shape:', session.outputs[n]));
  } catch (e) {
    console.error(e);
  }
}

async function run() {
  await check('../models/idr_tcn.onnx');
  await check('../newmodel/models/idr_tcn.onnx');
}

run();
