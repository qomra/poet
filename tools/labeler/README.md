# DiwanAI Server

A Node.js server for managing and adding AI-generated poetry responses to the DiwanAI application.

## Features

- Load and display poetry data from JSON files
- Add new AI responses to existing poems
- Real-time data updates
- RESTful API endpoints

## Setup

1. Install dependencies:
```bash
npm install
```

2. Make sure you have a `diwanai.json` file in the same directory with your poetry data.

3. Start the server:
```bash
npm start
```

For development with auto-restart:
```bash
npm run dev
```

## Usage

1. Open your browser and go to `http://localhost:3000`
2. Select a poem from the list
3. Click the "إضافة رد ذكاء اصطناعي جديد" (Add New AI Response) button
4. Fill in the AI response text, provider, and model
5. Click "إضافة" (Add) to save the new response

## API Endpoints

- `GET /api/poems` - Get all poems
- `POST /api/poems/:poemId/add-ai-response` - Add a new AI response to a poem

### Add AI Response Request Body:
```json
{
  "text": "AI generated poem text",
  "provider": "gemini",
  "model": "gemini-2.5-pro"
}
```

## Data Structure

The server expects a JSON file with the following structure:
```json
[
  {
    "poem_id": 123,
    "reference": {
      "poem": "Original poem text",
      "poet": "Poet name",
      "meter": "Poetic meter",
      "rhyme": "Rhyme scheme",
      "line_count": 10
    },
    "prompt": {
      "text": "Prompt used to generate AI response",
      "provider": "openai",
      "model": "gpt-4",
      "main_theme": "Theme",
      "tone": "Tone"
    },
    "ai": {
      "text": "AI generated poem",
      "provider": "gemini",
      "model": "gemini-2.5-pro"
    }
  }
]
```

## Requirements

- Node.js 14 or higher
- npm or yarn 