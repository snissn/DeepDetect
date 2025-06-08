import { useState } from 'react';
import ForensicsOverlay from '../components/ForensicsOverlay';

export default function Home() {
  const [src, setSrc] = useState(null);

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => setSrc(ev.target.result);
    reader.readAsDataURL(file);
  };

  return (
    <div className="container">
      <h1>Photo Forensics</h1>
      <p>Analyze your photos for signs of editing. Images never leave your device.</p>
      <input type="file" accept="image/*" onChange={handleFile} />
      {src && <ForensicsOverlay src={src} />}
    </div>
  );
}
