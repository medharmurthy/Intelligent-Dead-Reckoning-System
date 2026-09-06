const ort = require('onnxruntime-node');
const fs = require('fs');

async function runTest() {
  const modelPath = './public/models/idr_tcn_v2.onnx';
  const normPath = './public/data/normalization_v2.json';
  
  const normData = JSON.parse(fs.readFileSync(normPath, 'utf8'));
  
  const session = await ort.InferenceSession.create(modelPath);
  
  // Create synthetic stationary data
  const tensor = new Float32Array(180);
  for (let c = 0; c < 9; c++) {
    const mean = normData.mean[c];
    const std = normData.std[c];
    
    for (let t = 0; t < 20; t++) {
      let raw_val = 0;
      if (c === 2) raw_val = 9.8; // az
      if (c === 6) raw_val = 9.8; // accel_mag
      
      let norm_val = (raw_val - mean) / std;
      // apply clamp
      norm_val = Math.max(-5.0, Math.min(5.0, norm_val));
      
      tensor[c * 20 + t] = norm_val;
    }
  }

  const ortTensor = new ort.Tensor('float32', tensor, [1, 9, 20]);
  const feeds = { [session.inputNames[0]]: ortTensor };
  
  console.log("Running inference...");
  const results = await session.run(feeds);
  
  const speed = results['speed'].data[0];
  const logVar = results['log_var'].data[0];
  const variance = Math.exp(logVar);
  const logits = results['motion_logits'].data;
  
  console.log(`Speed Output: ${speed} m/s`);
  console.log(`Variance Output: ${variance}`);
  console.log(`Motion Logits:`, logits);

  if (speed < 0.5) {
      console.log("SUCCESS: New model predicts near 0 m/s for stationary data!");
  } else {
      console.warn("WARNING: Model still predicts non-stationary speeds for stationary data.");
  }
}

runTest().catch(console.error);
