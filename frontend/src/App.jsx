import React, { useState, useCallback, useRef, useEffect } from 'react';
import './App.css';
import { Upload, Activity, Volume2, Headphones, Lightbulb, FileAudio, AlertCircle, Download, Info, BarChart3, AudioWaveform, MessageSquare, Bot, BookOpen, Send } from 'lucide-react';
import Spectrogram3D from './components/Spectrogram3D.jsx';

const WaveformLogo = () => (
  <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="SoundMaster logo">
    <rect x="2" y="20" width="4" height="8" fill="#00FFE0"/>
    <rect x="9" y="14" width="4" height="20" fill="#00FFE0"/>
    <rect x="16" y="6" width="4" height="36" fill="#00FFE0"/>
    <rect x="23" y="10" width="4" height="28" fill="#00FFE0" opacity="0.9"/>
    <rect x="30" y="4" width="4" height="40" fill="#00FFE0"/>
    <rect x="37" y="16" width="4" height="16" fill="#00FFE0"/>
    <rect x="44" y="21" width="2" height="6" fill="#00FFE0"/>
  </svg>
);

const BACKEND_URL = import.meta.env.VITE_API_URL;
const API = `${BACKEND_URL}/api`;

const GLOSSARY = {
  composition: {
    title: "Composition",
    terms: [
      { term: "BPM", def: "Beats Per Minute - Tempo d'un morceau, nombre de battements par minute" },
      { term: "Mélodie", def: "Suite de notes formant une ligne musicale reconnaissable" },
      { term: "Harmonie", def: "Combinaison de notes jouées simultanément (accords)" },
      { term: "Progression d'accords", def: "Séquence d'accords qui forme la structure harmonique" },
      { term: "Hook", def: "Élément accrocheur et mémorable d'un morceau" },
      { term: "Bridge", def: "Section de transition entre couplet et refrain" },
      { term: "Drop", def: "Moment d'impact où tous les éléments entrent ensemble (EDM)" },
    ]
  },
  arrangement: {
    title: "Arrangement",
    terms: [
      { term: "Layering", def: "Superposition de sons pour créer de la richesse" },
      { term: "Build-up", def: "Montée en tension avant un drop ou refrain" },
      { term: "Breakdown", def: "Section épurée avec moins d'éléments" },
      { term: "Pad", def: "Nappe sonore qui remplit l'espace harmonique" },
      { term: "Arpège", def: "Notes d'un accord jouées successivement" },
      { term: "Contrepoint", def: "Lignes mélodiques indépendantes qui s'entrelacent" },
    ]
  },
  enregistrement: {
    title: "Enregistrement",
    terms: [
      { term: "Sample Rate", def: "Fréquence d'échantillonnage (44.1kHz, 48kHz, 96kHz)" },
      { term: "Bit Depth", def: "Résolution audio (16-bit, 24-bit, 32-bit)" },
      { term: "Gain staging", def: "Gestion des niveaux à chaque étape du signal" },
      { term: "DI", def: "Direct Input - Enregistrement direct sans micro" },
      { term: "Overhead", def: "Micros placés au-dessus (batterie notamment)" },
      { term: "Room mic", def: "Micro d'ambiance captant la réverbération naturelle" },
      { term: "Proximity effect", def: "Augmentation des basses quand le micro est proche" },
    ]
  },
  mixage: {
    title: "Mixage",
    terms: [
      { term: "EQ (Égalisation)", def: "Ajustement des fréquences d'un son" },
      { term: "Compression", def: "Réduction de la dynamique, contrôle des pics" },
      { term: "Ratio", def: "Taux de compression (ex: 4:1)" },
      { term: "Threshold", def: "Seuil à partir duquel le compresseur agit" },
      { term: "Attack/Release", def: "Temps de réaction du compresseur" },
      { term: "Reverb", def: "Simulation de l'acoustique d'un espace" },
      { term: "Delay", def: "Répétition retardée du signal (écho)" },
      { term: "Sidechain", def: "Déclenchement d'un effet par un autre signal" },
      { term: "Bus/Groupe", def: "Regroupement de pistes pour traitement commun" },
      { term: "Pan", def: "Positionnement gauche/droite dans l'image stéréo" },
      { term: "Send/Return", def: "Envoi vers un effet en parallèle" },
      { term: "High-pass filter (HPF)", def: "Filtre qui coupe les basses fréquences" },
      { term: "Low-pass filter (LPF)", def: "Filtre qui coupe les hautes fréquences" },
      { term: "Saturation", def: "Distorsion harmonique légère ajoutant de la chaleur" },
    ]
  },
  mastering: {
    title: "Mastering",
    terms: [
      { term: "LUFS", def: "Loudness Units Full Scale - Mesure de volume perçu (standard ITU-R BS.1770)" },
      { term: "True Peak", def: "Niveau de crête réel incluant l'inter-sample (dBTP)" },
      { term: "Headroom", def: "Marge entre le niveau max et 0 dBFS" },
      { term: "Limiter", def: "Compresseur avec ratio infini, empêche le clipping" },
      { term: "Dithering", def: "Bruit ajouté lors de la réduction de bit depth" },
      { term: "Mid/Side", def: "Traitement séparé du centre et des côtés stéréo" },
      { term: "Stereo Width", def: "Largeur de l'image stéréo" },
      { term: "Dynamic Range", def: "Écart entre les passages forts et faibles" },
      { term: "Clipping", def: "Distorsion quand le signal dépasse 0 dBFS" },
      { term: "Loudness War", def: "Course au volume excessif nuisant à la qualité" },
    ]
  },
  frequences: {
    title: "Fréquences",
    terms: [
      { term: "Sub-bass (20-60Hz)", def: "Graves profondes, ressenties plus qu'entendues" },
      { term: "Bass (60-250Hz)", def: "Fondamentales de la basse et du kick" },
      { term: "Low-mids (250-500Hz)", def: "Zone souvent boueuse, à traiter avec soin" },
      { term: "Mids (500Hz-2kHz)", def: "Corps des instruments, voix" },
      { term: "High-mids (2-4kHz)", def: "Présence, intelligibilité de la voix" },
      { term: "Highs (4-8kHz)", def: "Brillance, attaque des cymbales" },
      { term: "Air (8-20kHz)", def: "Souffle, ouverture, shimmer" },
    ]
  }
};

function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);

  // Chat — avec historique pour mémoire de conversation
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatEndRef = useRef(null);

  const [selectedCategory, setSelectedCategory] = useState('composition');

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleFileSelect = useCallback(async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const fileName = file.name.toLowerCase();
    const supported = ['.wav', '.mp3', '.flac', '.aiff', '.aif', '.ogg'];
    if (!supported.some(ext => fileName.endsWith(ext))) {
      setError('Format non supporté. Formats acceptés: WAV, MP3, FLAC, AIFF, OGG');
      return;
    }

    setSelectedFile(file.name);
    setError(null);
    setIsAnalyzing(true);
    setUploadProgress(0);
    setAnalysisResult(null);
    setActiveTab(0);

    try {
      const formData = new FormData();
      formData.append('file', file);
      setUploadProgress(5);

      const response = await fetch(`${API}/analyze-stream`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Erreur serveur ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete last line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const payload = JSON.parse(line.slice(6));
            if (payload.step === 'error') {
              throw new Error(payload.message || 'Erreur analyse');
            }
            if (typeof payload.progress === 'number') {
              setUploadProgress(payload.progress);
            }
            if (payload.step === 'done' && payload.result) {
              setAnalysisResult(payload.result);
            }
          } catch (parseErr) {
            if (parseErr.message !== 'Erreur analyse') console.warn('SSE parse:', parseErr);
            else throw parseErr;
          }
        }
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.message || 'Erreur inconnue');
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const sendChatMessage = useCallback(async () => {
    if (!chatInput.trim() || isChatLoading) return;

    const userMessage = chatInput.trim();
    setChatInput('');
    const newMessages = [...chatMessages, { role: 'user', content: userMessage }];
    setChatMessages(newMessages);
    setIsChatLoading(true);

    try {
      let context = null;
      if (analysisResult) {
        context = `Fichier: ${analysisResult.filename} | Score: ${analysisResult.overall_score}/100
LUFS: ${analysisResult.loudness_analysis.lufs_integrated} LUFS | True Peak: ${analysisResult.loudness_analysis.true_peak_db} dBTP
Headroom: ${analysisResult.loudness_analysis.headroom_db} dB | Dynamique: ${analysisResult.loudness_analysis.dynamic_range_db} dB`;
      }

      // Envoie l'historique complet pour que Claude se souvienne de la conversation
      const history = newMessages.slice(0, -1).map(m => ({ role: m.role, content: m.content }));

      const response = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage, context, history }),
      });
      const data = await response.json();

      setChatMessages([...newMessages, { role: 'assistant', content: data.response }]);
    } catch (err) {
      setChatMessages([...newMessages, {
        role: 'assistant',
        content: "Désolé, une erreur s'est produite. Réessayez."
      }]);
    } finally {
      setIsChatLoading(false);
    }
  }, [chatInput, isChatLoading, analysisResult, chatMessages]);

  const getScoreColor = (score) => {
    if (score >= 80) return '#4CAF50';
    if (score >= 60) return '#FFC107';
    if (score >= 40) return '#FF9800';
    return '#F44336';
  };

  const getDbColor = (db, type) => {
    if (type === 'headroom') {
      if (db >= 3) return '#4CAF50';
      if (db >= 1) return '#FFC107';
      return '#F44336';
    }
    if (type === 'dynamic') {
      if (db >= 8) return '#4CAF50';
      if (db >= 5) return '#FFC107';
      return '#FF9800';
    }
    if (type === 'truepeak') {
      return db > 0 ? '#F44336' : db > -1 ? '#FFC107' : '#4CAF50';
    }
    return '#64B5F6';
  };

  const getLufsColor = (lufs) => {
    // Standard streaming: autour de -14 LUFS
    if (lufs >= -16 && lufs <= -8) return '#4CAF50';
    if (lufs < -20 || lufs > -6) return '#F44336';
    return '#FFC107';
  };

  const handleExportPDF = useCallback(() => {
    if (analysisResult?.id) {
      window.open(`${API}/export-pdf/${analysisResult.id}`, '_blank');
    }
  }, [analysisResult]);

  const bandLabels = {
    'sub_bass': 'Sub Bass', 'bass': 'Bass', 'low_mids': 'Low Mids',
    'mids': 'Mids', 'high_mids': 'High Mids', 'highs': 'Highs', 'air': 'Air',
  };

  const renderFrequencyBar = (band, data) => {
    const normalizedEnergy = Math.max(0, Math.min(100, (data.energy_db + 60) * 1.5));
    return (
      <div key={band} className="frequency-row" data-testid={`freq-band-${band}`}>
        <span className="band-label">{bandLabels[band] || band}</span>
        <div className="bar-container">
          <div className="bar" style={{ width: `${normalizedEnergy}%` }} />
        </div>
        <span className="db-value">{data.energy_db.toFixed(1)} dB</span>
      </div>
    );
  };

  const tabs = [
    { icon: Info, label: 'Détails' },
    { icon: BarChart3, label: 'Spectral' },
    { icon: AudioWaveform, label: 'Audio' },
    { icon: MessageSquare, label: 'Conseils' },
    { icon: Bot, label: 'IA' },
    { icon: BookOpen, label: 'Index' },
  ];

  const renderDetailsTab = () => (
    <div className="tab-content" data-testid="tab-details">
      <div className="card score-card">
        <h2 className="section-title">Score Global</h2>
        <div className="score-circle">
          <span className="score-value" style={{ color: getScoreColor(analysisResult.overall_score) }}>
            {analysisResult.overall_score}
          </span>
          <span className="score-label">/100</span>
        </div>
      </div>

      <div className="card info-card">
        <h2 className="section-title">
          <FileAudio size={20} className="section-icon" />
          Informations du fichier
        </h2>
        <div className="info-row"><span className="info-label">Fichier:</span><span className="info-value">{analysisResult.filename}</span></div>
        <div className="info-row"><span className="info-label">Durée:</span><span className="info-value">{analysisResult.duration.toFixed(1)}s</span></div>
        <div className="info-row"><span className="info-label">Sample Rate:</span><span className="info-value">{analysisResult.sample_rate} Hz</span></div>
        <div className="info-row"><span className="info-label">Canaux:</span><span className="info-value">{analysisResult.channels === 1 ? 'Mono' : 'Stéréo'}</span></div>
      </div>

      <button className="export-pdf-btn" onClick={handleExportPDF} data-testid="export-pdf-btn">
        <Download size={20} />
        <span>Exporter le rapport PDF</span>
      </button>
    </div>
  );

  const renderSpectralTab = () => (
    <div className="tab-content" data-testid="tab-spectral">
      <div className="card analysis-card">
        <h2 className="section-title">
          <Activity size={20} className="section-icon" />
          Spectrogramme 3D
        </h2>
        {analysisResult.spectrogram_3d && <Spectrogram3D data={analysisResult.spectrogram_3d} />}
      </div>

      <div className="card analysis-card">
        <h2 className="section-title">
          <BarChart3 size={20} className="section-icon" />
          Analyseur de fréquences
        </h2>
        <div className="spectrum-visualizer">
          {Object.entries(analysisResult.frequency_analysis).map(([band, data]) => {
            const h = Math.max(8, Math.min(100, (data.energy_db + 60) * 1.5));
            const names = { 'sub_bass':'Sub','bass':'Bass','low_mids':'Low','mids':'Mid','high_mids':'Hi-M','highs':'High','air':'Air' };
            return (
              <div key={band} className="spectrum-bar-container">
                <div className="spectrum-bar-wrapper">
                  <div className="spectrum-bar" style={{ height: `${h}%` }} />
                </div>
                <span className="spectrum-label">{names[band]}</span>
                <span className="spectrum-db">{data.energy_db.toFixed(0)}</span>
              </div>
            );
          })}
        </div>
        <div className="frequency-details">
          {Object.entries(analysisResult.frequency_analysis).map(([band, data]) => renderFrequencyBar(band, data))}
        </div>
      </div>
    </div>
  );

  const renderAudioTab = () => (
    <div className="tab-content" data-testid="tab-audio">
      <div className="card analysis-card">
        <h2 className="section-title">
          <Volume2 size={20} className="section-icon" />
          Analyse Loudness
        </h2>
        <div className="metrics-grid">
          <div className="metric-box">
            <span className="metric-label">LUFS</span>
            <span className="metric-value" style={{ color: getLufsColor(analysisResult.loudness_analysis.lufs_integrated) }}>
              {analysisResult.loudness_analysis.lufs_integrated.toFixed(1)}
            </span>
            <span className="metric-sub">ITU-R BS.1770</span>
          </div>
          <div className="metric-box">
            <span className="metric-label">True Peak</span>
            <span className="metric-value" style={{ color: getDbColor(analysisResult.loudness_analysis.true_peak_db, 'truepeak') }}>
              {analysisResult.loudness_analysis.true_peak_db.toFixed(1)} dBTP
            </span>
            <span className="metric-sub">{analysisResult.loudness_analysis.true_peak_db > 0 ? '⚠️ Clipping' : '✓ OK'}</span>
          </div>
          <div className="metric-box">
            <span className="metric-label">Headroom</span>
            <span className="metric-value" style={{ color: getDbColor(analysisResult.loudness_analysis.headroom_db, 'headroom') }}>
              {analysisResult.loudness_analysis.headroom_db.toFixed(1)} dB
            </span>
          </div>
          <div className="metric-box">
            <span className="metric-label">Dynamique</span>
            <span className="metric-value" style={{ color: getDbColor(analysisResult.loudness_analysis.dynamic_range_db, 'dynamic') }}>
              {analysisResult.loudness_analysis.dynamic_range_db.toFixed(1)} dB
            </span>
          </div>
        </div>

        {/* Références LUFS */}
        <div className="lufs-references">
          <h3 className="lufs-ref-title">Références streaming</h3>
          {[
            { name: 'Spotify', target: -14 },
            { name: 'Apple Music', target: -16 },
            { name: 'YouTube', target: -14 },
            { name: 'Club/DJ', target: -7 },
          ].map(ref => {
            const diff = analysisResult.loudness_analysis.lufs_integrated - ref.target;
            return (
              <div key={ref.name} className="lufs-ref-row">
                <span className="lufs-ref-name">{ref.name}</span>
                <span className="lufs-ref-target">{ref.target} LUFS</span>
                <span className="lufs-ref-diff" style={{ color: Math.abs(diff) < 1 ? '#4CAF50' : Math.abs(diff) < 3 ? '#FFC107' : '#F44336' }}>
                  {diff > 0 ? `+${diff.toFixed(1)}` : diff.toFixed(1)} dB
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="card analysis-card">
        <h2 className="section-title">
          <Headphones size={20} className="section-icon" />
          Image Stéréo
        </h2>
        {analysisResult.stereo_analysis.is_stereo ? (
          <div className="stereo-grid">
            <div className="stereo-item">
              <span className="stereo-label">Corrélation {analysisResult.stereo_analysis.phase_issues ? '⚠️' : ''}</span>
              <span className="stereo-value" style={{ color: analysisResult.stereo_analysis.phase_issues ? '#F44336' : '#4CAF50' }}>
                {(analysisResult.stereo_analysis.correlation * 100).toFixed(0)}%
              </span>
              <div className="stereo-bar-bg">
                <div className="stereo-bar-fill" style={{
                  width: `${Math.abs(analysisResult.stereo_analysis.correlation) * 100}%`,
                  backgroundColor: analysisResult.stereo_analysis.correlation > 0.3 ? '#4CAF50' : '#F44336'
                }} />
              </div>
              {analysisResult.stereo_analysis.phase_issues && (
                <span className="phase-warning">Problème de phase détecté</span>
              )}
            </div>
            <div className="stereo-item">
              <span className="stereo-label">Largeur Stéréo</span>
              <span className="stereo-value">{(analysisResult.stereo_analysis.width * 100).toFixed(0)}%</span>
              <div className="stereo-bar-bg">
                <div className="stereo-bar-fill" style={{ width: `${analysisResult.stereo_analysis.width * 100}%`, backgroundColor: '#6C63FF' }} />
              </div>
            </div>
            <div className="stereo-item">
              <span className="stereo-label">Balance L/R</span>
              <span className="stereo-value">
                {analysisResult.stereo_analysis.balance > 0 ? 'R ' : analysisResult.stereo_analysis.balance < 0 ? 'L ' : ''}
                {Math.abs(analysisResult.stereo_analysis.balance * 100).toFixed(0)}%
              </span>
              <div className="balance-bar-bg">
                <div className="balance-center" />
                <div className="balance-fill" style={{
                  left: analysisResult.stereo_analysis.balance < 0 ? `${50 + analysisResult.stereo_analysis.balance * 50}%` : '50%',
                  width: `${Math.abs(analysisResult.stereo_analysis.balance) * 50}%`,
                }} />
              </div>
            </div>
          </div>
        ) : (
          <p className="mono-note">{analysisResult.stereo_analysis.note || "Fichier mono"}</p>
        )}
      </div>
    </div>
  );

  const renderAdviceTab = () => (
    <div className="tab-content" data-testid="tab-advice">
      <div className="card recommendations-card">
        <h2 className="section-title">
          <Lightbulb size={20} className="section-icon recommendation-icon" />
          Recommandations IA
        </h2>
        <p className="advice-intro">Basé sur l'analyse de votre fichier, voici les corrections suggérées :</p>
        {analysisResult.recommendations.map((rec, index) => (
          <div key={index} className="recommendation-item" data-testid={`recommendation-${index}`}>
            <div className="rec-number"><span>{index + 1}</span></div>
            <p className="recommendation-text">{rec}</p>
          </div>
        ))}
      </div>
    </div>
  );

  const renderChatTab = () => (
    <div className="tab-content" data-testid="tab-chat">
      <div className="card chat-card">
        <h2 className="section-title">
          <Bot size={20} className="section-icon" />
          Assistant Ingénieur Son
        </h2>
        <p className="chat-intro">
          Posez vos questions sur le mixage, mastering, ou demandez des explications sur l'analyse.
        </p>

        <div className="chat-messages">
          {chatMessages.length === 0 && (
            <div className="chat-welcome">
              <Bot size={40} className="chat-welcome-icon" />
              <p>Bonjour ! Je suis votre assistant ingénieur du son. Comment puis-je vous aider ?</p>
              <div className="chat-suggestions">
                <button onClick={() => setChatInput("Comment améliorer mon mix ?")}>Comment améliorer mon mix ?</button>
                <button onClick={() => setChatInput("Explique-moi le sidechain")}>Explique-moi le sidechain</button>
                <button onClick={() => setChatInput("Quels plugins recommandes-tu ?")}>Quels plugins recommandes-tu ?</button>
              </div>
            </div>
          )}

          {chatMessages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.role}`}>
              <div className="chat-message-content">{msg.content}</div>
            </div>
          ))}

          {isChatLoading && (
            <div className="chat-message assistant">
              <div className="chat-message-content typing">
                <span></span><span></span><span></span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="chat-input-container">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
            placeholder="Posez votre question..."
            className="chat-input"
            disabled={isChatLoading}
          />
          <button onClick={sendChatMessage} className="chat-send-btn" disabled={isChatLoading || !chatInput.trim()}>
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );

  const renderIndexTab = () => (
    <div className="tab-content" data-testid="tab-index">
      <div className="card glossary-card">
        <h2 className="section-title">
          <BookOpen size={20} className="section-icon" />
          Glossaire Audio
        </h2>
        <p className="glossary-intro">Tous les termes de la production musicale, de la composition au mastering.</p>

        <div className="glossary-categories">
          {Object.keys(GLOSSARY).map((cat) => (
            <button key={cat} className={`glossary-cat-btn ${selectedCategory === cat ? 'active' : ''}`} onClick={() => setSelectedCategory(cat)}>
              {GLOSSARY[cat].title}
            </button>
          ))}
        </div>

        <div className="glossary-terms">
          <h3 className="glossary-section-title">{GLOSSARY[selectedCategory].title}</h3>
          {GLOSSARY[selectedCategory].terms.map((item, index) => (
            <div key={index} className="glossary-item">
              <dt className="glossary-term">{item.term}</dt>
              <dd className="glossary-def">{item.def}</dd>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <div className="app" data-testid="soundmaster-app">
      <div className="content">
        <header className="header" data-testid="app-header">
          <img src="/logo.png" alt="SoundMaster" className="header-logo" />
          <p className="subtitle">Analyseur Audio Professionnel</p>
        </header>

        <label className={`upload-button ${isAnalyzing ? 'disabled' : ''}`} data-testid="upload-button">
          <input
            type="file"
            accept=".wav,.mp3,.flac,.aiff,.aif,.ogg,audio/wav,audio/mpeg,audio/flac,audio/aiff,audio/ogg"
            onChange={handleFileSelect}
            disabled={isAnalyzing}
            style={{ display: 'none' }}
            data-testid="file-input"
          />
          {isAnalyzing ? (
            <div className="uploading-container">
              <div className="spinner" />
              <span className="uploading-text">Analyse en cours... {uploadProgress}%</span>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
              </div>
            </div>
          ) : (
            <>
              <Upload size={48} className="upload-icon" />
              <span className="upload-text">Télécharger un fichier audio</span>
              <span className="upload-subtext">WAV · MP3 · FLAC · AIFF · OGG</span>
            </>
          )}
        </label>

        {error && (
          <div className="error-message" data-testid="error-message">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        )}

        {selectedFile && !isAnalyzing && !error && (
          <div className="selected-file" data-testid="selected-file">
            <FileAudio size={16} />
            <span>{selectedFile}</span>
          </div>
        )}

        {analysisResult && (
          <div className="results-container" data-testid="results-container">
            <div className="tabs-nav" data-testid="tabs-nav">
              {tabs.map((tab, index) => (
                <button key={index} className={`tab-btn ${activeTab === index ? 'active' : ''}`}
                  onClick={() => setActiveTab(index)} data-testid={`tab-btn-${index}`}>
                  <tab.icon size={18} />
                  <span>{tab.label}</span>
                </button>
              ))}
            </div>
            <div className="tab-panel">
              {activeTab === 0 && renderDetailsTab()}
              {activeTab === 1 && renderSpectralTab()}
              {activeTab === 2 && renderAudioTab()}
              {activeTab === 3 && renderAdviceTab()}
              {activeTab === 4 && renderChatTab()}
              {activeTab === 5 && renderIndexTab()}
            </div>
          </div>
        )}

        {!analysisResult && !isAnalyzing && (
          <div className="empty-state" data-testid="empty-state">
            <p className="empty-text">
              Téléchargez un fichier audio pour obtenir une analyse professionnelle avec recommandations IA.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
