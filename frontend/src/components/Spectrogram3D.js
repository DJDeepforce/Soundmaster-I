import React from 'react';
import Plot from 'react-plotly.js';

function Spectrogram3D({ data }) {
  if (!data || !data.amplitudes || data.amplitudes.length === 0) {
    return (
      <div className="spectrogram-3d-placeholder">
        <p>Chargement du spectrogramme...</p>
      </div>
    );
  }

  // Prepare data for Plotly surface plot
  const z = data.amplitudes.map(row => row.map(val => val * 10)); // Scale amplitude
  
  // Create frequency labels (log scale for better visualization)
  const freqLabels = data.frequencies.map(f => {
    if (f >= 1000) return `${(f/1000).toFixed(1)}k`;
    return `${Math.round(f)}`;
  });
  
  // Create time labels
  const timeLabels = data.times.map(t => t.toFixed(2));

  return (
    <div className="spectrogram-3d-container" data-testid="spectrogram-3d">
      <Plot
        data={[
          {
            type: 'surface',
            z: z,
            x: timeLabels,
            y: freqLabels,
            colorscale: [
              [0, 'rgb(20, 0, 40)'],
              [0.1, 'rgb(60, 0, 100)'],
              [0.2, 'rgb(100, 0, 150)'],
              [0.3, 'rgb(150, 0, 180)'],
              [0.4, 'rgb(200, 50, 150)'],
              [0.5, 'rgb(255, 100, 100)'],
              [0.6, 'rgb(255, 150, 50)'],
              [0.7, 'rgb(255, 200, 0)'],
              [0.8, 'rgb(200, 255, 50)'],
              [0.9, 'rgb(100, 255, 100)'],
              [1, 'rgb(50, 255, 200)']
            ],
            showscale: false,
            lighting: {
              ambient: 0.6,
              diffuse: 0.8,
              specular: 0.3,
              roughness: 0.5,
            },
            lightposition: {
              x: 100,
              y: 200,
              z: 100
            },
            contours: {
              z: {
                show: true,
                usecolormap: true,
                highlightcolor: "#fff",
                project: { z: false }
              }
            }
          }
        ]}
        layout={{
          autosize: true,
          height: 350,
          margin: { l: 0, r: 0, t: 30, b: 0 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          scene: {
            xaxis: {
              title: 'Temps (s)',
              titlefont: { color: '#888', size: 10 },
              tickfont: { color: '#666', size: 8 },
              gridcolor: 'rgba(108, 99, 255, 0.2)',
              showbackground: true,
              backgroundcolor: 'rgba(10, 10, 20, 0.8)'
            },
            yaxis: {
              title: 'Fréquence (Hz)',
              titlefont: { color: '#888', size: 10 },
              tickfont: { color: '#666', size: 8 },
              gridcolor: 'rgba(108, 99, 255, 0.2)',
              showbackground: true,
              backgroundcolor: 'rgba(10, 10, 20, 0.8)'
            },
            zaxis: {
              title: 'Amplitude',
              titlefont: { color: '#888', size: 10 },
              tickfont: { color: '#666', size: 8 },
              gridcolor: 'rgba(108, 99, 255, 0.2)',
              showbackground: true,
              backgroundcolor: 'rgba(10, 10, 20, 0.8)'
            },
            camera: {
              eye: { x: 1.5, y: 1.5, z: 1.2 },
              center: { x: 0, y: 0, z: -0.2 }
            },
            aspectmode: 'manual',
            aspectratio: { x: 2, y: 1.5, z: 0.8 }
          }
        }}
        config={{
          displayModeBar: true,
          modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
          displaylogo: false,
          scrollZoom: true
        }}
        style={{ width: '100%', height: '350px' }}
        useResizeHandler={true}
      />
      <p className="spectrogram-3d-hint">Glissez pour faire pivoter • Scroll pour zoomer</p>
    </div>
  );
}

export default Spectrogram3D;
