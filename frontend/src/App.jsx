import React, { useState } from "react";
import axios from "axios";

const API_URL = "http://localhost:5000";

function App() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [imageName, setImageName] = useState("");
  const [probs, setProbs] = useState(null);

  const handleFileChange = (e) => {
    setImage(e.target.files[0]);
    setPrediction("");
    setImageName("");
    setProbs(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;

    const formData = new FormData();
    formData.append("image", image);

    try {
      const res = await axios.post(`${API_URL}/api/clasificar`, formData);
      const data = res.data;
      setPrediction(data.prediction);
      setImageName(data.image_name);
      setProbs(data.probs);
    } catch (err) {
      alert("Error al clasificar la imagen");
      console.error(err);
    }
  };

  const imageUrl = imageName ? `${API_URL}/uploads/${imageName}` : "";
  const graphUrl = imageName
    ? `${API_URL}/uploads/probabilidades.png?t=${new Date().getTime()}`
    : "";

  return (
    <div className="app">
      <header>
        <h1>Clasificador de Tumores Cerebrales</h1>
        <form onSubmit={handleSubmit}>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            required
          />
          <button type="submit">Clasificar Imagen</button>
        </form>
      </header>

      <main className="content">
        <section className="left">
          {imageUrl && (
            <div className="card">
              <h2>Imagen Analizada</h2>
              <img src={imageUrl} alt="MRI" className="preview-img" />
            </div>
          )}
          {prediction && (
            <div className="card result-card">
              <h2>Resultado</h2>
              <p className="prediction">{prediction}</p>
            </div>
          )}
        </section>

        <section className="right">
          {probs && (
            <div className="card">
              <h2>Gráfico de Probabilidades</h2>
              <img src={graphUrl} alt="Gráfico" className="chart-img" />
            </div>
          )}
          {probs && (
            <div className="card">
              <h2>Detalles por Clase</h2>
              <table>
                <thead>
                  <tr>
                    <th>Clase</th>
                    <th>Probabilidad</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(probs).map(([clase, prob]) => (
                    <tr key={clase}>
                      <td>{clase}</td>
                      <td>{(prob * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;