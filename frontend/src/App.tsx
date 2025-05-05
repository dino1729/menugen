import React, { useState, useRef, useEffect } from 'react';
import './App.css';

interface MenuItem {
  name: string;
  description?: string;
  imageUrl?: string;
  error?: string;
  // Add category if available from backend, otherwise handle grouping differently
  category?: string;
}

// Helper: make sure “Spaghetti ”, “spaghetti”, “SPAGHETTI” all map to a single key
const normalizeName = (s: string) => s.trim().toLowerCase();

// Convert whatever the backend sends into something the browser can load
const BACKEND_BASE_URL = 'http://localhost:8000';
const buildImageUrl = (raw: string): string => {
  if (raw.startsWith('data:')) return raw; // base-64 already usable
  if (raw.startsWith('http://') || raw.startsWith('https://')) return raw; // already absolute
  // If relative, prepend backend base URL
  if (raw.startsWith('/')) return BACKEND_BASE_URL + raw;
  return BACKEND_BASE_URL + '/' + raw;
};

// Fallback loader: download image yourself and turn it into an object-URL.
const downloadAsObjectURL = async (remoteUrl: string): Promise<string> => {
  try {
    console.log('[FETCH] start →', remoteUrl);
    const res = await fetch(remoteUrl, { mode: 'cors' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    const objUrl = URL.createObjectURL(blob);
    console.log('[FETCH] success → objectURL:', objUrl);
    return objUrl;
  } catch (e) {
    console.error('[FETCH] FAILED', e);
    throw e;
  }
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>('');
  const [menuItems, setMenuItems] = useState<MenuItem[]>([]);
  const [images, setImages] = useState<{ [key: string]: string }>({});
  // Track individual image loading status for fade-in effect
  const [imageLoadStatus, setImageLoadStatus] = useState<{[key: string]: 'loading'|'loaded'|'error'}>({});
  const [errors, setErrors] = useState<string[]>([]);
  const [isParseOnly, setIsParseOnly] = useState<boolean>(false);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null); // To display the uploaded menu image
  const [progress, setProgress] = useState<number>(0); // For progress bar
  const [totalItems, setTotalItems] = useState<number>(0); // Total items for progress calculation
  const wsRef = useRef<WebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null); // Ref for the file input
  // New: Track if generation is in progress (WebSocket open and not done)
  const [isGenerating, setIsGenerating] = useState(false);

  // Modal state for full image view
  const [modalOpen, setModalOpen] = useState(false);
  const [modalImage, setModalImage] = useState<string | null>(null);
  const [modalDesc, setModalDesc] = useState<string>('');
  const [modalTitle, setModalTitle] = useState<string>('');

  const openImageModal = (imgUrl: string, desc: string, title: string) => {
    setModalImage(imgUrl);
    setModalDesc(desc);
    setModalTitle(title);
    setModalOpen(true);
  };
  const closeModal = () => setModalOpen(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const selectedFile = event.target.files[0];
      setFile(selectedFile);
      // Create a URL for the uploaded image preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedImageUrl(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
      // Reset state when a new file is selected
      resetState(false); // Don't clear the file/image preview
    }
  };

  const resetState = (clearFile = true) => {
    setStatus('');
    setMenuItems([]);
    setImages({});
    setErrors([]);
    setIsParseOnly(false);
    setProgress(0);
    setTotalItems(0);
    if (clearFile) {
        setFile(null);
        setUploadedImageUrl(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = ""; // Clear the file input visually
        }
    }
    if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
    }
  }

  const handleUploadAndGenerate = async () => {
    if (!file) return;
    resetState(false); // Keep file, clear results
    setIsParseOnly(false);
    setStatus('Preparing to process...'); // Always show status immediately
    setProgress(2); // Show a little progress right away
    const formData = new FormData();
    formData.append('file', file);
    try {
      setStatus('Uploading menu...');
      setProgress(5); // Initial progress
      const response = await fetch('http://localhost:8000/upload_menu/', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setStatus('Extracting menu items...');
      setProgress(15);
      if (data.sessionId) {
        connectWebSocket(data.sessionId);
      } else {
        setStatus('Failed to start processing session.');
        setErrors([data.message || 'Unknown error starting session.']);
        setProgress(0);
      }
    } catch (error) {
      setStatus('Upload failed');
      setErrors([String(error)]);
      setProgress(0);
    }
  };

  const handleParseOnly = async () => {
    if (!file) return;
    resetState(false); // Keep file, clear results
    setIsParseOnly(true);
    setStatus('Preparing to process...'); // Always show status immediately
    setProgress(2);
    const formData = new FormData();
    formData.append('file', file);
    try {
      setStatus('Uploading and Parsing...');
      setProgress(10);
      const response = await fetch('http://localhost:8000/parse_menu_only/', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok && data.status === 'success') {
        const items = data.data || [];
        setMenuItems(items.map((item: any) => ({ name: item.name, description: item.description, category: item.section || 'Uncategorized' })));
        setStatus('Menu parsed successfully.');
        setProgress(100);
      } else {
        setStatus('Parsing failed');
        setErrors([data.message || 'Unknown error during parsing.']);
        setProgress(0);
      }
    } catch (error) {
      setStatus('Parsing request failed');
      setErrors([String(error)]);
      setProgress(0);
    }
  };

  const handleStopGenerating = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setIsGenerating(false);
      setStatus('Generation stopped by user');
    }
  };

  const handleRefresh = () => {
    resetState(true);
    setIsGenerating(false);
  };

  const connectWebSocket = (sessionId: string) => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
    wsRef.current = ws;
    setIsGenerating(true);
    ws.onopen = () => {
        setStatus('Connected. Waiting for results...');
        setProgress(20); // Progress update on connect
    }
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      console.log("WebSocket message received:", msg); // Log all incoming messages
      switch (msg.type) {
        case 'status':
          setStatus(msg.message);
          // Progress mapping: parse status message
          if (msg.message.startsWith('Parsing menu')) {
            setProgress(15);
          } else {
            const match = msg.message.match(/Generating image for .*\((\d+)\/(\d+)\)/);
            if (match) {
              const i = parseInt(match[1], 10);
              const total = parseInt(match[2], 10);
              setProgress(30 + (i / total) * 70);
            }
          }
          break;
        case 'menu_parsed':
          // Fix: Handle both dict and list for msg.data
          const items = Array.isArray(msg.data) ? msg.data : (msg.data.items || []);
          setMenuItems(items.map((item: any) => ({ 
            name: item.name, 
            description: item.description, 
            category: item.section || 'Uncategorized' 
          })));
          setTotalItems(items.length); // Set total items for progress calculation
          // Show generation start immediately
          setStatus(`Generating images (0/${items.length})`);
          setProgress(30); // Progress after parsing
          break;
        case 'image_generated': {
          const rawName = msg.item;
          const key = normalizeName(rawName);
          const fallback = async (origUrl: string) => {
            try {
              const blobUrl = await downloadAsObjectURL(origUrl);
              // store blob URL – will definitely load
              setImages(prev => ({ ...prev, [key]: blobUrl }));
              setImageLoadStatus(prev => ({ ...prev, [key]: 'loaded' }));
            } catch {
              setImageLoadStatus(prev => ({ ...prev, [key]: 'error' }));
              setErrors(prev => [...prev, `Could not fetch image for ${rawName}`]);
            }
          };

          // mark loading
          setImageLoadStatus(prev => ({ ...prev, [key]: 'loading' }));

          const finalUrl = buildImageUrl(msg.url);
          console.log(`[IMG] direct attempt  key=«${key}»  url=`, finalUrl);

          // Save the direct URL first
          setImages(prev => ({ ...prev, [key]: finalUrl }));

          // also try pre-fetching so we know early if the URL is bad
          downloadAsObjectURL(finalUrl)
            .then(() => console.log(`[CHECK] ${finalUrl} reachable`))
            .catch(() => {
              console.warn(`[CHECK] ${finalUrl} unreachable – switching to blob fallback`);
              fallback(finalUrl);
            });

          // ...existing progress calculation...
          break;
        }
        case 'image_error':
          setMenuItems(prev => prev.map(item => item.name === msg.item ? { ...item, error: msg.message } : item));
          setErrors((prev) => [...prev, `Image error for ${msg.item}: ${msg.message}`]);
          setImages((prev) => { // Still count error as "processed" for progress
            const newImages = { ...prev, [msg.item]: 'error' }; // Mark as error processed
            const processedCount = Object.keys(newImages).length;
            const imageProgress = totalItems > 0 ? (processedCount / totalItems) * 70 : 0;
            setProgress(30 + imageProgress);
            return newImages;
          });
          break;
        case 'error':
          setErrors((prev) => [...prev, `Error: ${msg.message}`]);
          setStatus('Error occurred during processing');
          setProgress(0); // Reset progress on critical error
          setIsGenerating(false);
          break;
        case 'done':
          setStatus('All images generated!');
          setProgress(100);
          setIsGenerating(false);
          
          // Debug log the final images state
          console.log("Final images state:", images);
          break;
        default:
          console.log("Unknown WS message type:", msg.type);
      }
    };
    ws.onerror = (error) => {
        console.error("WebSocket Error:", error);
        setStatus('WebSocket error');
        setErrors((prev) => [...prev, 'WebSocket connection error.']);
        setProgress(0);
        setIsGenerating(false);
    };
    ws.onclose = (event) => {
        console.log("WebSocket Closed:", event.reason, event.code);
        // Update status only if processing wasn't finished or errored out
        if (progress < 100 && !status.toLowerCase().includes('error') && !status.toLowerCase().includes('failed')) {
             setStatus('WebSocket closed unexpectedly');
             // Optionally reset progress or leave as is
        }
        setIsGenerating(false);
    };
  };

  // Add useEffect to log images whenever they change
  useEffect(() => {
    console.log("Images state updated (relative URLs):", images);
    console.log("Total image count:", Object.keys(images).length);
  }, [images]);

  // Extra logging to see when image load-status changes
  useEffect(() => {
    console.log('imageLoadStatus changed ➜', imageLoadStatus);
  }, [imageLoadStatus]);
  
  // Group items by category for rendering
  const groupedMenuItems = menuItems.reduce((acc, item) => {
    const category = item.category || 'Uncategorized';
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(item);
    return acc;
  }, {} as { [key: string]: MenuItem[] });

  return (
    <div className="App">
      <header className="App-header">
        <div className="logo">MenuGen</div>
        <div className="header-actions">
          {/* Placeholder for settings/user icons */}
          <span className="settings-icon">⚙️</span>
          <span className="user-icon">👤</span>
        </div>
      </header>

      <main className="App-main">
        {/* Dedicated Status Bar at the top of the app */}
        <div className="global-status-bar">
          Status: {status || 'Idle'}
          {progress > 0 && progress < 100 && (
            <span style={{ marginLeft: 12, color: '#888', fontSize: '0.98em' }}>({Math.round(progress)}%)</span>
          )}
        </div>
        <h1>Turn Menus into <span className="magic-text">Magic</span></h1>
        <p className="subtitle">Upload any menu and watch as AI transforms each dish into stunning, mouth-watering visuals. ✨🍽️</p>

        {!file && (
          <div className="upload-area">
            <input
              type="file"
              id="fileUpload"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/png, image/jpeg, image/gif, application/pdf" // Accept images and PDFs
              style={{ display: 'none' }} // Hide the default input
            />
            {/* Use label to make the whole area clickable */}
            <label htmlFor="fileUpload" className="upload-box">
              <div className="upload-content"> {/* Added wrapper for centering */}
                <div className="upload-icon">☁️</div>
                <p>Click to upload or drag and drop</p>
                <p className="upload-hint">PNG, JPG, GIF up to 10MB</p>
              </div>
            </label>
          </div>
        )}

        {file && (
          <div className="processing-area">
            <h2>Uploaded Menu Image</h2>
            {uploadedImageUrl && <img src={uploadedImageUrl} alt="Uploaded Menu" className="uploaded-menu-image" />}

            {/* Progress Bar (lavender color) */}
            {(progress > 0 && progress < 100) && !isParseOnly && (
              <div className="progress-container lavender-bar">
                <div className="progress-bar-bg">
                  <div className="progress-bar-fg" style={{ width: `${progress}%` }}></div>
                </div>
                <p className="status-bar-text">{Math.round(progress)}%</p>
                <div className="status-bar-text">Status: {status || 'Idle'}</div>
              </div>
            )}
            {/* Progress Bar for Parse Only */}
            {(progress > 0 && progress < 100) && isParseOnly && (
              <div className="progress-container lavender-bar">
                <div className="progress-bar-bg">
                  <div className="progress-bar-fg" style={{ width: `${progress}%` }}></div>
                </div>
                <p className="status-bar-text">{Math.round(progress)}%</p>
                <div className="status-bar-text">Status: {status || 'Idle'}</div>
              </div>
            )}
            {/* Show status bar at 100% (done) */}
            {progress === 100 && (
              <div className="progress-container lavender-bar">
                <div className="progress-bar-bg">
                  <div className="progress-bar-fg" style={{ width: `100%` }}></div>
                </div>
                <p className="status-bar-text">All done!</p>
                <div className="status-bar-text">Status: Done</div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="action-buttons">
              <button onClick={handleParseOnly} disabled={!file || status.startsWith('Uploading') || status.startsWith('Processing') || status.startsWith('Connected') || status.startsWith('Generating')}>Parse Only</button>
              <button onClick={handleUploadAndGenerate} disabled={!file || status.startsWith('Uploading') || status.startsWith('Processing') || status.startsWith('Connected') || status.startsWith('Generating')}>Parse & Generate Images</button>
              <button onClick={() => resetState(true)} className="new-menu-button">Upload New Menu</button>
              {/* Stop Generating Button */}
              <button
                onClick={handleStopGenerating}
                className="stop-generating-button"
                style={{ background: '#e53935', color: '#fff' }}
                disabled={!isGenerating}
              >
                Stop Generating
              </button>
              {/* Refresh Button */}
              <button
                onClick={handleRefresh}
                className="refresh-button"
                style={{ background: '#ff9100', color: '#fff' }}
                title="Refresh (hard reset)"
              >
                &#x1F3D5; Refresh
              </button>
            </div>
          </div>
        )}

        {errors.length > 0 && (
          <div className="error-container">
            <strong>Errors:</strong>
            <ul>
              {errors.map((e, i) => <li key={i}>{e}</li>)}
            </ul>
          </div>
        )}

        <div className="results-area">
          {Object.keys(groupedMenuItems).length > 0 && (
            <div>
              <h2>{isParseOnly ? 'Parsed Menu Items' : 'Menu Items & Generated Images'}</h2>
              {/* Debug section removed */}
              {Object.entries(groupedMenuItems).map(([category, items]) => (
                <section key={category} className="menu-category">
                  <h3 className="category-title">{category}</h3>
                  <div className="menu-items-grid">
                    {items.map(item => {
                      const key = normalizeName(item.name);
                      const imageUrl = images[key];
                      const hasImage = imageUrl && imageUrl !== 'error';
                      return (
                        <div key={item.name} className="menu-item-card">
                          {!isParseOnly && hasImage && (
                            <div className="menu-item-image-container">
                              <img
                                src={buildImageUrl(imageUrl)}
                                alt={item.name}
                                className={`menu-item-image ${imageLoadStatus[key] === 'loaded' ? 'loaded' : ''}`}
                                style={{ maxHeight: '220px', objectFit: 'cover', cursor: 'pointer' }}
                                onClick={() => openImageModal(buildImageUrl(imageUrl), item.description || '', item.name)}
                                onLoad={() => {
                                  setImageLoadStatus(prev => ({ ...prev, [key]: 'loaded' }));
                                }}
                                onError={(e) => {
                                  const direct = buildImageUrl(images[key]);
                                  downloadAsObjectURL(direct)
                                    .then(blobUrl => {
                                      setImages(prev => ({ ...prev, [key]: blobUrl }));
                                    })
                                    .catch(() => {
                                      setImageLoadStatus(prev => ({ ...prev, [key]: 'error' }));
                                    });
                                }}
                              />
                            </div>
                          )}
                          <div className="menu-item-details">
                            <strong className="menu-item-name">{item.name}</strong>
                            <p className="menu-item-description">
                              {item.description && item.description.trim() !== ''
                                ? item.description
                                : <span style={{ color: '#999', fontStyle: 'italic' }}>No description provided</span>}
                            </p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </section>
              ))}
            </div>
          )}
        </div>
      </main>
      {/* Modal for full image view */}
      {modalOpen && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={closeModal}>&times;</button>
            <img src={modalImage!} alt={modalTitle} className="modal-full-image" />
            <div className="modal-desc-area">
              <div className="modal-title">{modalTitle}</div>
              <div className="modal-desc">{modalDesc || <span style={{ color: '#999', fontStyle: 'italic' }}>No description provided</span>}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
