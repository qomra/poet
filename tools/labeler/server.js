const express = require('express');
const fs = require('fs').promises;
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('.'));

// Data file path
const DATA_FILE = 'diwanai.json';

// Load data from JSON file
async function loadData() {
    try {
        const data = await fs.readFile(DATA_FILE, 'utf8');
        const parsedData = JSON.parse(data);
        
        // Add virtual ID to each item
        return parsedData.map((item, index) => ({
            ...item,
            virtualId: index
        }));
    } catch (error) {
        console.error('Error loading data:', error);
        return [];
    }
}

// Save data to JSON file
async function saveData(data) {
    try {
        // Remove virtualId before saving
        const dataToSave = data.map(({ virtualId, ...item }) => item);
        await fs.writeFile(DATA_FILE, JSON.stringify(dataToSave, null, 2), 'utf8');
        return true;
    } catch (error) {
        console.error('Error saving data:', error);
        return false;
    }
}

// Routes
app.get('/api/poems', async (req, res) => {
    try {
        const data = await loadData();
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to load poems' });
    }
});

app.post('/api/poems/:poemId/add-ai-response', async (req, res) => {
    try {
        const { poemId } = req.params;
        const { text, provider = 'gemini', model = 'gemini-2.5-pro', thoughtMap } = req.body;

        if (!text) {
            return res.status(400).json({ error: 'Text is required' });
        }

        const data = await loadData();
        const originalPoem = data.find(poem => poem.poem_id == poemId);

        if (!originalPoem) {
            return res.status(404).json({ error: 'Poem not found' });
        }

        // Create new poem with new AI response
        const newPoem = {
            ...originalPoem,
            ai: {
                text: text,
                provider: provider,
                model: model,
                ...(thoughtMap && { thoughtMap: thoughtMap })
            }
        };

        // Add to data
        data.push(newPoem);

        // Save to file
        const saved = await saveData(data);
        if (!saved) {
            return res.status(500).json({ error: 'Failed to save data' });
        }

        res.json({ 
            success: true, 
            newPoem: newPoem,
            message: 'AI response added successfully' 
        });

    } catch (error) {
        console.error('Error adding AI response:', error);
        res.status(500).json({ error: 'Failed to add AI response' });
    }
});

app.delete('/api/poems/:virtualId', async (req, res) => {
    try {
        const { virtualId } = req.params;
        const data = await loadData();
        
        const itemIndex = data.findIndex(item => item.virtualId == virtualId);
        
        if (itemIndex === -1) {
            return res.status(404).json({ error: 'Item not found' });
        }

        // Remove the item
        data.splice(itemIndex, 1);
        
        // Reassign virtual IDs
        data.forEach((item, index) => {
            item.virtualId = index;
        });

        // Save to file
        const saved = await saveData(data);
        if (!saved) {
            return res.status(500).json({ error: 'Failed to save data' });
        }

        res.json({ 
            success: true, 
            message: 'Item deleted successfully' 
        });

    } catch (error) {
        console.error('Error deleting item:', error);
        res.status(500).json({ error: 'Failed to delete item' });
    }
});

// Serve the HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'diwanai.html'));
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
