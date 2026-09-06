import fs from 'fs';

const data = fs.readFileSync('data/raw/IO-VNBD/dataset1.csv', 'utf8');
const lines = data.split('\n').slice(1, 100);

for (const line of lines) {
  if (!line.trim()) continue;
  const cols = line.split(',');
  const time = cols[0];
  const speed = parseFloat(cols[3]); // Assuming speed is col 3
  const ay = parseFloat(cols[5]); // Assuming ay is col 5
  console.log(`Speed: ${speed}, ay: ${ay}`);
}
