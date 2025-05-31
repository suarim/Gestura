import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import "./App.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

const LANGS = [
  { code: "en", label: "English" },
  { code: "hi", label: "Hindi" },
  { code: "es", label: "Spanish" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
  { code: "zh-cn", label: "Chinese" },
  { code: "ja", label: "Japanese" },
  { code: "ru", label: "Russian" },
  { code: "pt", label: "Portuguese" },
  { code: "bn", label: "Bengali" },
  { code: "pa", label: "Punjabi" },
  { code: "ur", label: "Urdu" },
  { code: "vi", label: "Vietnamese" },
  { code: "it", label: "Italian" }
];

function App() {
  const webcamRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [translation, setTranslation] = useState("");
  const [animatedWords, setAnimatedWords] = useState([]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [showTranslation, setShowTranslation] = useState(false);
  const [copyButtonText, setCopyButtonText] = useState("Copy");
  const [targetLang, setTargetLang] = useState("en");
  const [translatedText, setTranslatedText] = useState("");
  const [voices, setVoices] = useState([]);
  const captureIntervalRef = useRef(null);
  const [langOpen, setLangOpen] = useState(false);

  const currentLangLabel = LANGS.find((l) => l.code === targetLang)?.label || "Lang";

  const animateWords = (text) => {
    if (!text || text.trim() === "") {
      setAnimatedWords([]);
      setIsAnimating(false);
      setShowTranslation(false);
      return;
    }

    const words = text.trim().split(/\s+/);
    
    // Reset states and prepare for animation
    setAnimatedWords([]);
    setIsAnimating(true);
    setShowTranslation(true);

    words.forEach((word, index) => {
      setTimeout(() => {
        setAnimatedWords(prev => [...prev, word]);
        
        // Stop animating after the last word
        if (index === words.length - 1) {
          setTimeout(() => setIsAnimating(false), 100);
        }
      }, index * 200); // 200ms delay between words
    });
  };

  const copyToClipboard = async () => {
    if (!translation) return;
    
    try {
      await navigator.clipboard.writeText(translation);
      setCopyButtonText("Copied!");
      setTimeout(() => setCopyButtonText("Copy"), 2000);
    } catch (err) {
      console.error("Failed to copy text:", err);
    }
  };

  const translateText = async () => {
    if (targetLang === "en") {
      // English requested – reset to original
      setTranslatedText("");
      // return;
    }
    try {
      // setTargetLang("en")
      const resp = await fetch(`${BACKEND_URL}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: translation, target_lang: targetLang }),
      });
      const data = await resp.json();
      if (data.translated_text) {
        setTranslatedText(data.translated_text);
        animateWords(data.translated_text);
      }
    } catch (err) {
      console.error("Translation failed", err);
    }
  };

  const startRecording = async () => {
    setTranslation("");
    setAnimatedWords([]);
    setIsAnimating(false);
    setShowTranslation(false);
    setCopyButtonText("Copy");
    setTargetLang("en")
    setTranslatedText("");
    try {
      await fetch(`${BACKEND_URL}/start`, { method: "POST" });
      setRecording(true);
    } catch (err) {
      alert("Failed to start recording: " + err);
    }
  };

  const stopRecording = async () => {
    setRecording(false);
    clearInterval(captureIntervalRef.current);
    captureIntervalRef.current = null;

    try {
      const resp = await fetch(`${BACKEND_URL}/stop`, { method: "POST" });
      const data = await resp.json();
      const newTranslation = data.translation || "No translation returned.";
      setTranslation(newTranslation);
      
      // Start word animation immediately, no delay
      animateWords(newTranslation);
    } catch (err) {
      alert("Failed to stop recording: " + err);
    }
  };

  // Capture frames at ~3 FPS while recording
  useEffect(() => {
    if (recording) {
      captureIntervalRef.current = setInterval(async () => {
        if (!webcamRef.current) return;
        const imageSrc = webcamRef.current.getScreenshot();
        if (!imageSrc) return;
        try {
          await fetch(`${BACKEND_URL}/frame`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageSrc }),
          });
        } catch (err) {
          console.error("Failed to send frame", err);
        }
      }, 333); // ~3 frames per second
    }

    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
    };
  }, [recording]);

  // Load speech synthesis voices
  useEffect(() => {
    const loadVoices = () => {
      const v = window.speechSynthesis.getVoices();
      if (v.length) setVoices(v);
    };
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;

    const videoElement = webcamRef.current?.video;
  }, []);

  const speakText = () => {
    const utterance = new SpeechSynthesisUtterance(translatedText || translation);
    // find best matching voice
    const match = voices.find((v) => v.lang.toLowerCase().startsWith(targetLang.toLowerCase()));
    if (match) utterance.voice = match;
    else if (voices.length) utterance.voice = voices[0];
    window.speechSynthesis.speak(utterance);
  };

  const renderTranslation = () => {
    if (!showTranslation) return null;
  
    return (
      <div className="translation-text">
        {animatedWords.map((word, index) => (
          <span
            key={index}
            className="word"
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            {word}&nbsp;
          </span>
        ))}
        {isAnimating && <span className="translation-cursor"></span>}
      </div>
    );
  };
  

  // Create floating particles
  const renderParticles = () => {
    return (
      <div className="particles">
        {[...Array(1000)].map((_, i) => (
          <div key={i} className="particle"></div>
        ))}
      </div>
    );
  };

  return (
    <>
      <header className="app-header">
        <div className="header-particle"></div>
        <div className="header-particle"></div>
        <div className="header-particle"></div>
        <div className="header-particle"></div>
        
        <div className="header-left">
          <div className="app-logo"></div>
        </div>
        
        <div className="header-center">
          <div className="title-decoration"></div>
          <h1 className="app-title">gestura</h1>
          <div className="app-subtitle">Sign Language Translation</div>
        </div>
        
        <div className="header-right">
          <div className={`status-indicator ${recording ? 'recording' : ''}`}>
            <div className="status-dot"></div>
            <span>{recording ? 'Recording' : 'Ready'}</span>
          </div>
          <div className="version-badge">v1.0</div>
        </div>
      </header>
    
      <div className="container">
        {renderParticles()}
      
        <div className="main-content">
          <div className="left-panel">
            <div className="webcam-section">
              <div className="webcam-container">
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  style={{ width: "100%", height: "100%" ,transform: "scaleX(-1)"}}
                />
                {recording && (
                  <div className="recording-indicator">
                    <div className="recording-dot"></div>
                    Recording
                  </div>
                )}
              </div>
              
              <div className="controls">
                {recording ? (
                  <button onClick={stopRecording} className="control-button stop-button">
                    Stop Recording
                  </button>
                ) : (
                  <button onClick={startRecording} className="control-button start-button">
                    Start Recording
                  </button>
                )}
              </div>
            </div>
          </div>
          
          <div className="right-panel">
            <h2 className="translation-header">Translation</h2>
            <div className="translation-container">
              <div className="translation-output">
                {renderTranslation()}
              </div>
              <button 
                onClick={copyToClipboard}
                className={`copy-button ${copyButtonText === "Copied!" ? "copied" : ""}`}
                disabled={!translation}
                title={translation ? "Copy translation" : "No translation to copy"}
              >
                {copyButtonText}
              </button>
              {/* Translate controls */}
              <div style={{ marginTop: "1rem", display: "flex", gap: "0.5rem", alignItems: "center" }}>
                {/* Custom dropdown */}
                <div className="lang-dropdown">
                  <button
                    type="button"
                    className="lang-trigger"
                    onClick={() => setLangOpen((o) => !o)}
                  >
                    {currentLangLabel}
                    <span style={{ transform: langOpen ? "rotate(180deg)" : "none", transition: "transform 0.2s" }}>▾</span>
                  </button>
                  {langOpen && (
                    <div className="lang-menu" onMouseLeave={() => setLangOpen(false)}>
                      {LANGS.map((l) => (
                        <div
                          key={l.code}
                          className="lang-item"
                          onClick={() => {
                            setTargetLang(l.code);
                            setLangOpen(false);
                          }}
                        >
                          {l.label}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  onClick={translateText}
                  className="control-button start-button"
                  style={{ padding: "8px 16px", fontSize: "14px" }}
                  disabled={!translation}
                >
                  Translate
                </button>
                <button
                  onClick={speakText}
                  className="control-button start-button"
                  style={{ padding: "8px 16px", fontSize: "14px" }}
                  disabled={!(translatedText || translation)}
                >
                  Speak
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default App; 