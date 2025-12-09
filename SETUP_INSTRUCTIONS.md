# Bug Classification UI - Setup Instructions

This React app has been migrated from Create React App to **Vite** for a lighter, faster development experience with fewer dependencies.

## Prerequisites

- Node.js 16+ installed
- Backend API running on http://localhost:5000

## Installation Steps

### Step 1: Clean Install

```bash
# Delete node_modules if it exists
rmdir /s /q node_modules     # Windows CMD
# or
Remove-Item -Recurse -Force node_modules   # PowerShell

# Install dependencies
npm install
```

### Step 2: Run Development Server

```bash
npm run dev
```

The app will open at **http://localhost:3000**

### Step 3: Build for Production

```bash
npm run build
```

The production build will be in the `dist` folder.

## Key Changes from Create React App

- ✅ **Lighter dependencies** - No `react-scripts`, no `flatted.py` issues
- ✅ **Faster startup** - Vite is significantly faster than webpack
- ✅ **Simpler config** - Just `vite.config.js`
- ✅ **Same React code** - Your components work exactly the same

## Troubleshooting

### If npm install still fails on Windows:

1. **Try with administrator privileges:**
   ```bash
   # Right-click PowerShell -> Run as Administrator
   npm install
   ```

2. **Move project out of OneDrive folder**

3. **Use Yarn instead:**
   ```bash
   npm install -g yarn
   yarn install
   yarn dev
   ```

4. **Or open the standalone `standalone.html`** - No installation needed!

## Project Structure

```
pythontest/
├── index.html          # Entry HTML file (Vite uses root, not public/)
├── vite.config.js      # Vite configuration
├── package.json        # Dependencies
├── src/
│   ├── index.jsx       # React entry point
│   ├── App.jsx         # Main App component
│   └── App.css         # Minimal global styles
└── standalone.html     # No-build version (backup)
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## Backend Connection

The app connects to the backend API at `http://localhost:5000`. Make sure your Python backend is running:

```bash
cd backend
python backend_api.py
```

---

**For corporate environments with strict permissions**, the `standalone.html` file provides a no-build, no-install alternative that loads everything from CDN.
