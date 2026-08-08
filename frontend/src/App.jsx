import { useEffect, useState } from "react";
import { generateFrames } from "./api";

function isImageFile(file) {
  return file && (file.type.startsWith("image/") || /\.(png|jpe?g)$/i.test(file.name));
}

function FrameInput({ label, file, onChange }) {
  const [previewUrl, setPreviewUrl] = useState("");

  useEffect(() => {
    if (!isImageFile(file)) {
      setPreviewUrl("");
      return undefined;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  return (
    <label className="file-input">
      <span>{label}</span>
      <input type="file" accept=".npy,.png,.jpg,.jpeg,image/png,image/jpeg" onChange={(event) => onChange(event.target.files?.[0] ?? null)} />
      <small>{file ? file.name : "Choose a .npy, PNG, or JPEG satellite frame"}</small>
      {previewUrl && <img className="upload-preview" src={previewUrl} alt={`${label} preview`} />}
      {file && !previewUrl && <small className="npy-note">NumPy files are processed by the backend; preview is unavailable before generation.</small>}
    </label>
  );
}

export default function App() {
  const [frame0, setFrame0] = useState(null);
  const [frame1, setFrame1] = useState(null);
  const [count, setCount] = useState(1);
  const [status, setStatus] = useState("");
  const [results, setResults] = useState([]);
  const [gifUrl, setGifUrl] = useState("");

  async function submit(event) {
    event.preventDefault();
    if (!frame0 || !frame1) {
      setStatus("Choose both satellite-frame files first.");
      return;
    }
    setStatus("Generating interpolated frames…");
    setResults([]);
    setGifUrl("");
    try {
      const response = await generateFrames(frame0, frame1, count);
      setResults(response.frames);
      setGifUrl(response.gif_url);
      setStatus(
        `Created ${response.frames.length} intermediate frame(s).` +
        (response.resized_input ? " The later image was resized to match the earlier image." : ""),
      );
    } catch (error) {
      setStatus(error.message);
    }
  }

  return (
    <main>
      <section className="hero">
        <p className="eyebrow">SATELLITE FRAME INTERPOLATION</p>
        <h1>ChronoCloud</h1>
        <p>Generate cloud-motion frames between two geostationary satellite scans.</p>
      </section>

      <form onSubmit={submit} className="card">
        <div className="inputs">
          <FrameInput label="Earlier frame (.npy, PNG, JPEG)" file={frame0} onChange={setFrame0} />
          <FrameInput label="Later frame (.npy, PNG, JPEG)" file={frame1} onChange={setFrame1} />
        </div>
        <label className="select-input">
          <span>Intermediate frames</span>
          <select value={count} onChange={(event) => setCount(Number(event.target.value))}>
            <option value="1">1 frame — midpoint</option>
            <option value="3">3 frames — quarter intervals</option>
            <option value="7">7 frames — eighth intervals</option>
          </select>
        </label>
        <button type="submit">Generate frames</button>
        {status && <p className="status" role="status">{status}</p>}
      </form>

      {results.length > 0 && (
        <section className="results">
          <h2>Generated frames</h2>
          <article className="animation-card">
            <img src={gifUrl} alt="Looping animation of generated satellite frames" />
            <div>
              <h3>Animation</h3>
              <p>Looping GIF created from the generated intermediate frames.</p>
              <a href={gifUrl} download>Download GIF</a>
            </div>
          </article>
          <div className="result-grid">
            {results.map((frame) => (
              <article key={frame.index} className="result-card">
                <img src={frame.preview_url} alt={`Generated satellite frame ${frame.index}`} />
                <p>Intermediate frame {frame.index}</p>
                <a href={frame.npy_url} download>Download .npy</a>
              </article>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
