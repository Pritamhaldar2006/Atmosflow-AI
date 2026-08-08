const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function generateFrames(frame0, frame1, numIntermediate) {
  const payload = new FormData();
  payload.append("frame0", frame0);
  payload.append("frame1", frame1);
  const response = await fetch(
    `${API_URL}/interpolate?num_intermediate=${numIntermediate}`,
    { method: "POST", body: payload },
  );
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail ?? "Generation failed.");
  return {
    ...body,
    gif_url: `${API_URL}${body.gif_url}`,
    frames: body.frames.map((frame) => ({
      ...frame,
      preview_url: `${API_URL}${frame.preview_url}`,
      npy_url: `${API_URL}${frame.npy_url}`,
    })),
  };
}
