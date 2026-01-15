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

// Define constants for backend communication
const EFFECTIVE_HOSTNAME = window.location.hostname;
const BACKEND_PORT = '8005';

// NVIDIA-specific image generation models
const NVIDIA_IMAGE_MODELS = [
  { value: 'stabilityai/stable-diffusion-3.5-large', label: 'Stable Diffusion 3.5 Large' },
  { value: 'black-forest-labs/flux.1-kontext-dev', label: 'FLUX.1 Kontext Dev' },
  { value: 'black-forest-labs/flux.1-schnell', label: 'FLUX.1 Schnell' },
  { value: 'black-forest-labs/flux.1-dev', label: 'FLUX.1 Dev' },
  { value: 'stabilityai/stable-diffusion-3-medium', label: 'Stable Diffusion 3 Medium' },
];

const API_ORIGIN = `${window.location.protocol}//${EFFECTIVE_HOSTNAME}:${BACKEND_PORT}`;
const WS_ORIGIN = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${EFFECTIVE_HOSTNAME}:${BACKEND_PORT}`;

// Convert whatever the backend sends into something the browser can load
const BACKEND_BASE_URL = API_ORIGIN; // Use the consistent API_ORIGIN for image paths

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
  // Theme state - default to 'light'
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    // Check local storage for saved theme preference
    const savedTheme = localStorage.getItem('menugen-theme');
    // Return saved theme or default to 'light'
    return (savedTheme === 'dark' ? 'dark' : 'light') as 'light' | 'dark';
  });

  // Theme toggle function
  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    // Save to local storage
    localStorage.setItem('menugen-theme', newTheme);
  };

  // Configuration and model selection state
  const [showSettings, setShowSettings] = useState(false);
  const [imageProvider, setImageProvider] = useState<string>('litellm');
  const [visionModel, setVisionModel] = useState<string>('gpt-4o');
  const [imageGenModel, setImageGenModel] = useState<string>('gemini-3-pro-image-preview');
  const [videoGenModel, setVideoGenModel] = useState<string>('veo-3.1-generate-001');
  const [descriptionModel, setDescriptionModel] = useState<string>('gemini-3-flash-preview');

  // Available models from backend (curated whitelists from config.json)
  const [availableModels, setAvailableModels] = useState<{
    vision: string[];
    image: string[];
    video: string[];
    text: string[];
  }>({
    vision: [],
    image: [],
    video: [],
    text: []
  });

  // Fetch configuration and curated model whitelists on mount
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch(`${API_ORIGIN}/config`);
        if (response.ok) {
          const config = await response.json();
          // Set current model selections from backend defaults
          setImageProvider(config.image_provider || 'litellm');
          setVisionModel(config.vision_model || 'gpt-4o');
          setImageGenModel(config.image_gen_model || 'gemini-3-pro-image-preview');
          setVideoGenModel(config.video_gen_model || 'veo-3.1-generate-001');
          setDescriptionModel(config.description_model || 'gemini-3-flash-preview');

          // Use curated whitelists from config.json if available
          if (config.whitelists) {
            setAvailableModels({
              vision: config.whitelists.vision || [],
              text: config.whitelists.text || [],
              image: config.whitelists.image || [],
              video: config.whitelists.video || []
            });
          }
        }
      } catch (error) {
        console.error('Failed to fetch config:', error);
      }
    };

    fetchConfig();
  }, []);

  // Update image generation model when provider changes
  useEffect(() => {
    if (imageProvider === 'nvidia') {
      // Set to first NVIDIA model if current model is not a NVIDIA model
      const isCurrentModelNvidia = NVIDIA_IMAGE_MODELS.some(m => m.value === imageGenModel);
      if (!isCurrentModelNvidia) {
        setImageGenModel(NVIDIA_IMAGE_MODELS[0].value);
      }
    } else if (imageProvider === 'litellm' && availableModels.image.length > 0) {
      // Set to first available LiteLLM image model if current is an NVIDIA model
      const isCurrentModelNvidia = NVIDIA_IMAGE_MODELS.some(m => m.value === imageGenModel);
      if (isCurrentModelNvidia) {
        setImageGenModel(availableModels.image[0]);
      }
    }
  }, [imageProvider, availableModels.image]);

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
    formData.append('image_provider', imageProvider);
    formData.append('vision_model', visionModel);
    formData.append('image_gen_model', imageGenModel);
    formData.append('video_gen_model', videoGenModel);
    formData.append('description_model', descriptionModel);
    try {
      setStatus(`Uploading menu (using ${imageProvider.toUpperCase()} with ${imageGenModel})...`);
      setProgress(5); // Initial progress
      const response = await fetch(`${API_ORIGIN}/upload_menu/`, {
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
    formData.append('vision_model', visionModel);
    formData.append('description_model', descriptionModel);
    try {
      setStatus(`Uploading and Parsing with ${visionModel}...`);
      setProgress(10);
      const response = await fetch(`${API_ORIGIN}/parse_menu_only/`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok && data.status === 'success') {
        // Fix: Handle both dict and list for data.data
        const rawItems = Array.isArray(data.data) ? data.data : (data.data?.items);

        if (Array.isArray(rawItems)) {
          const validItems = rawItems
            .filter((item: any): item is { name: any; description?: any; section?: any } => 
              item && typeof item === 'object' && typeof item.name === 'string'
            )
            .map((item: { name: any; description?: any; section?: any }) => ({
              name: String(item.name), // Ensure name is a string
              description: typeof item.description === 'string' ? item.description : '', // Default if not a string
              category: typeof item.section === 'string' ? item.section : 'Uncategorized', // Default if not a string
            }));
          setMenuItems(validItems);
        } else {
          // If rawItems is not an array (e.g., null, undefined, or other type), treat as no items
          setMenuItems([]);
          console.warn("Parsed data.data.items was not an array:", rawItems);
        }
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
    const ws = new WebSocket(`${WS_ORIGIN}/ws/${sessionId}`);
    wsRef.current = ws;
    setIsGenerating(true);
    ws.onopen = () => {
        setStatus('Connected. Waiting for results...');
        setProgress(20); // Progress update on connect
    }
    ws.onmessage = (event: MessageEvent) => {
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
              setImages((prev: { [key: string]: string }) => ({ ...prev, [key]: blobUrl }));
              setImageLoadStatus((prev: { [key: string]: 'loading'|'loaded'|'error' }) => ({ ...prev, [key]: 'loaded' }));
            } catch {
              setImageLoadStatus((prev: { [key: string]: 'loading'|'loaded'|'error' }) => ({ ...prev, [key]: 'error' }));
              setErrors((prev: string[]) => [...prev, `Could not fetch image for ${rawName}`]);
            }
          };

          // mark loading
          setImageLoadStatus((prev: { [key: string]: 'loading'|'loaded'|'error' }) => ({ ...prev, [key]: 'loading' }));

          const finalUrl = buildImageUrl(msg.url);
          console.log(`[IMG] direct attempt  key=«${key}»  url=`, finalUrl);

          // Save the direct URL first
          setImages((prev: { [key: string]: string }) => ({ ...prev, [key]: finalUrl }));

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
          setMenuItems((prev: MenuItem[]) => prev.map((item: MenuItem) => item.name === msg.item ? { ...item, error: msg.message } : item));
          setErrors((prev: string[]) => [...prev, `Image error for ${msg.item}: ${msg.message}`]);
          setImages((prev: { [key: string]: string }) => { // Still count error as "processed" for progress
            const newImages = { ...prev, [msg.item]: 'error' }; // Mark as error processed
            const processedCount = Object.keys(newImages).length;
            const imageProgress = totalItems > 0 ? (processedCount / totalItems) * 70 : 0;
            setProgress(30 + imageProgress);
            return newImages;
          });
          break;
        case 'error':
          setErrors((prev: string[]) => [...prev, `Error: ${msg.message}`]);
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
        setErrors((prev: string[]) => [...prev, 'WebSocket connection error.']);
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
  const groupedMenuItems = menuItems.reduce((acc: { [key: string]: MenuItem[] }, item: MenuItem) => {
    const category = item.category || 'Uncategorized';
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(item);
    return acc;
  }, {} as { [key: string]: MenuItem[] });

  // Add useEffect to apply the theme to the document body when theme changes
  useEffect(() => {
    document.body.setAttribute('data-theme', theme);
  }, [theme]);

  return (
    <div className={`App ${theme}-theme`}>
      <header className="App-header">
        <div className="logo">MenuGen</div>
        
        <div className="header-actions">
          <button 
            onClick={() => setShowSettings(true)}
            className="settings-toggle"
            title="Configure Models & Providers"
            aria-label="Open settings to configure models and providers"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12,8A4,4 0 0,1 16,12A4,4 0 0,1 12,16A4,4 0 0,1 8,12A4,4 0 0,1 12,8M12,10A2,2 0 0,0 10,12A2,2 0 0,0 12,14A2,2 0 0,0 14,12A2,2 0 0,0 12,10M10,22C9.75,22 9.54,21.82 9.5,21.58L9.13,18.93C8.5,18.68 7.96,18.34 7.44,17.94L4.95,18.95C4.73,19.03 4.46,18.95 4.34,18.73L2.34,15.27C2.21,15.05 2.27,14.78 2.46,14.63L4.57,12.97L4.5,12L4.57,11L2.46,9.37C2.27,9.22 2.21,8.95 2.34,8.73L4.34,5.27C4.46,5.05 4.73,4.96 4.95,5.05L7.44,6.05C7.96,5.66 8.5,5.32 9.13,5.07L9.5,2.42C9.54,2.18 9.75,2 10,2H14C14.25,2 14.46,2.18 14.5,2.42L14.87,5.07C15.5,5.32 16.04,5.66 16.56,6.05L19.05,5.05C19.27,4.96 19.54,5.05 19.66,5.27L21.66,8.73C21.79,8.95 21.73,9.22 21.54,9.37L19.43,11L19.5,12L19.43,13L21.54,14.63C21.73,14.78 21.79,15.05 21.66,15.27L19.66,18.73C19.54,18.95 19.27,19.04 19.05,18.95L16.56,17.95C16.04,18.34 15.5,18.68 14.87,18.93L14.5,21.58C14.46,21.82 14.25,22 14,22H10M11.25,4L10.88,6.61C9.68,6.86 8.62,7.5 7.85,8.39L5.44,7.35L4.69,8.65L6.8,10.2C6.4,11.37 6.4,12.64 6.8,13.8L4.68,15.36L5.43,16.66L7.86,15.62C8.63,16.5 9.68,17.14 10.87,17.38L11.24,20H12.76L13.13,17.39C14.32,17.14 15.37,16.5 16.14,15.62L18.57,16.66L19.32,15.36L17.2,13.81C17.6,12.64 17.6,11.37 17.2,10.2L19.31,8.65L18.56,7.35L16.15,8.39C15.38,7.5 14.32,6.86 13.12,6.62L12.75,4H11.25Z" />
            </svg>
            <span className="model-label">{imageProvider.toUpperCase()}</span>
          </button>
          <button 
            onClick={toggleTheme}
            className="theme-toggle"
            title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
            aria-label={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
          >
            {theme === 'light' ? (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                <path d="M9.37,5.51C9.19,6.15,9.1,6.82,9.1,7.5c0,4.08,3.32,7.4,7.4,7.4c0.68,0,1.35-0.09,1.99-0.27C17.45,17.19,14.93,19,12,19 c-3.86,0-7-3.14-7-7C5,9.07,6.81,6.55,9.37,5.51z M12,3c-4.97,0-9,4.03-9,9s4.03,9,9,9s9-4.03,9-9c0-0.46-0.04-0.92-0.1-1.36 c-0.98,1.37-2.58,2.26-4.4,2.26c-2.98,0-5.4-2.42-5.4-5.4c0-1.81,0.89-3.42,2.26-4.4C12.92,3.04,12.46,3,12,3L12,3z"/>
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                <path d="M12,7c-2.76,0-5,2.24-5,5s2.24,5,5,5s5-2.24,5-5S14.76,7,12,7L12,7z M2,13h2c0.55,0,1-0.45,1-1s-0.45-1-1-1H2 c-0.55,0-1,0.45-1,1S1.45,13,2,13z M20,13h2c0.55,0,1-0.45,1-1s-0.45-1-1-1h-2c-0.55,0-1,0.45-1,1S19.45,13,20,13z M11,2v2 c0,0.55,0.45,1,1,1s1-0.45,1-1V2c0-0.55-0.45-1-1-1S11,1.45,11,2z M11,20v2c0,0.55,0.45,1,1,1s1-0.45,1-1v-2c0-0.55-0.45-1-1-1 S11,19.45,11,20z M5.99,4.58c-0.39-0.39-1.03-0.39-1.41,0c-0.39,0.39-1.03,0.39,1.41,0l1.06,1.06c0.39,0.39,1.03,0.39,1.41,0 s0.39-1.03,0-1.41L5.99,4.58z M18.36,16.95c-0.39-0.39-1.03-0.39-1.41,0c-0.39,0.39-0.39,1.03,0,1.41l1.06,1.06 c0.39,0.39,1.03,0.39,1.41,0c0.39-0.39,0.39-1.03,0-1.41L18.36,16.95z M19.42,5.99c0.39-0.39,0.39-1.03,0-1.41 c-0.39-0.39-1.03-0.39-1.41,0l-1.06,1.06c-0.39,0.39-0.39,1.03,0,1.41s1.03,0.39,1.41,0L19.42,5.99z M7.05,18.36 c0.39-0.39,0.39-1.03,0-1.41c-0.39-0.39-1.03-0.39-1.41,0l-1.06,1.06c-0.39,0.39-0.39,1.03,0,1.41s1.03,0.39,1.41,0L7.05,18.36z"/>
              </svg>
            )}
          </button>
        </div>
      </header>

      <main className="App-main">
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
                <div className="status-bar-text status">{status || 'Processing...'}</div>
              </div>
            )}
            {/* Progress Bar for Parse Only */}
            {(progress > 0 && progress < 100) && isParseOnly && (
              <div className="progress-container lavender-bar">
                <div className="progress-bar-bg">
                  <div className="progress-bar-fg" style={{ width: `${progress}%` }}></div>
                </div>
                <p className="status-bar-text">{Math.round(progress)}%</p>
                <div className="status-bar-text status">{status || 'Processing...'}</div>
              </div>
            )}
            {/* Show status bar at 100% (done) */}
            {progress === 100 && (
              <div className="progress-container lavender-bar">
                <div className="progress-bar-bg">
                  <div className="progress-bar-fg" style={{ width: `100%` }}></div>
                </div>
                <p className="status-bar-text"><span className="done-text">All done!</span> ✨</p>
                <div className="status-bar-text status">{status || 'Done'}</div>
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
              {errors.map((e: string, i: number) => <li key={i}>{e}</li>)}
            </ul>
          </div>
        )}

        <div className="results-area">
          {Object.keys(groupedMenuItems).length > 0 && (
            <div>
              <h2>{isParseOnly ? 'Parsed Menu Items' : 'Menu Items & Generated Images'}</h2>
              {/* Debug section removed */}
              {Object.entries(groupedMenuItems).map(([category, items]: [string, MenuItem[]]) => (
                <section key={category} className="menu-category">
                  <h3 className="category-title">{category}</h3>
                  <div className="menu-items-grid">
                    {items.map((item: MenuItem) => {
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
                                  setImageLoadStatus((prev: { [key: string]: 'loading'|'loaded'|'error' }) => ({ ...prev, [key]: 'loaded' }));
                                }}
                                onError={() => {
                                  setImageLoadStatus((prev: { [key: string]: 'loading'|'loaded'|'error' }) => ({ ...prev, [key]: 'error' }));
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
              <div className="modal-desc">{modalDesc || <span style={{ opacity: 0.6, fontStyle: 'italic' }}>No description provided</span>}</div>
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {showSettings && (
        <div className="modal-overlay" onClick={() => setShowSettings(false)}>
          <div className="settings-modal" onClick={e => e.stopPropagation()}>
            <div className="settings-header">
              <h2>Model Settings</h2>
              <button className="modal-close" onClick={() => setShowSettings(false)}>&times;</button>
            </div>
            
            <div className="settings-body">
              <div className="setting-group">
                <label>Image Provider</label>
                <select
                  value={imageProvider}
                  onChange={(e) => setImageProvider(e.target.value)}
                  className="setting-select"
                >
                  <option value="litellm">LiteLLM (Proxy)</option>
                  <option value="nvidia">NVIDIA Direct</option>
                </select>
                <span className="setting-hint">Provider for image generation (NVIDIA requires API key)</span>
              </div>

              <div className="setting-group">
                <label>Vision Model (Menu Parsing)</label>
                <select 
                  value={visionModel} 
                  onChange={(e) => setVisionModel(e.target.value)}
                  className="setting-select"
                >
                  {availableModels.vision.length > 0 ? (
                    availableModels.vision.map(model => (
                      <option key={model} value={model}>{model}</option>
                    ))
                  ) : (
                    <option value={visionModel}>{visionModel}</option>
                  )}
                </select>
                <span className="setting-hint">Model for understanding menu images</span>
              </div>

              <div className="setting-group">
                <label>Image Generation Model</label>
                <select
                  value={imageGenModel}
                  onChange={(e) => setImageGenModel(e.target.value)}
                  className="setting-select"
                >
                  {imageProvider === 'nvidia' ? (
                    NVIDIA_IMAGE_MODELS.map(model => (
                      <option key={model.value} value={model.value}>{model.label}</option>
                    ))
                  ) : availableModels.image.length > 0 ? (
                    availableModels.image.map(model => (
                      <option key={model} value={model}>{model}</option>
                    ))
                  ) : (
                    <option value={imageGenModel}>{imageGenModel}</option>
                  )}
                </select>
                <span className="setting-hint">Model for generating menu item images</span>
              </div>

              <div className="setting-group">
                <label>Video Generation Model</label>
                <select 
                  value={videoGenModel} 
                  onChange={(e) => setVideoGenModel(e.target.value)}
                  className="setting-select"
                >
                  {availableModels.video.length > 0 ? (
                    availableModels.video.map(model => (
                      <option key={model} value={model}>{model}</option>
                    ))
                  ) : (
                    <option value={videoGenModel}>{videoGenModel}</option>
                  )}
                </select>
                <span className="setting-hint">Model for video generation (future use)</span>
              </div>

              <div className="setting-group">
                <label>Description Model</label>
                <select
                  value={descriptionModel}
                  onChange={(e) => setDescriptionModel(e.target.value)}
                  className="setting-select"
                >
                  {availableModels.text.length > 0 ? (
                    availableModels.text.map(model => (
                      <option key={model} value={model}>{model}</option>
                    ))
                  ) : (
                    <option value={descriptionModel}>{descriptionModel}</option>
                  )}
                </select>
                <span className="setting-hint">Model for generating/simplifying descriptions</span>
              </div>
            </div>

            <div className="settings-footer">
              <button 
                className="btn-primary" 
                onClick={() => setShowSettings(false)}
              >
                Apply Settings
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
