from flask import Flask, request, render_template, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
from io import BytesIO
import logging
import sqlite3
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['DATABASE'] = 'gallery.db'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and processor
processor = None
model = None


def get_db():
    """Get database connection"""
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database with schema"""
    db = get_db()
    cursor = db.cursor()
    
    # Create albums table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS albums (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create images table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            album_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            caption TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (album_id) REFERENCES albums (id) ON DELETE CASCADE
        )
    ''')
    
    db.commit()
    db.close()
    logger.info("Database initialized")

# ============== FILE HANDLING ==============


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_album_folder(album_id):
    """Get folder path for album"""
    return os.path.join(app.config['UPLOAD_FOLDER'], f'album_{album_id}')


def ensure_album_folder(album_id):
    """Ensure album folder exists"""
    folder = get_album_folder(album_id)
    os.makedirs(folder, exist_ok=True)
    return folder


def load_model():
    """Load the BLIP model and processor"""
    global processor, model
    try:
        logger.info("Loading BLIP model...")
        
        # Load processor
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        if processor is None:
            raise Exception("Failed to load processor")
        logger.info("Processor loaded successfully")
        
        # Load model
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        if model is None:
            raise Exception("Failed to load model")
        logger.info("Model loaded successfully")
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        # Verify model is properly loaded
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Processor type: {type(processor)}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        processor = None
        model = None
        raise


def generate_caption(image_path):
    """Generate caption for an image"""
    global model, processor
    
    try:
        # Check if model and processor are loaded
        if model is None or processor is None:
            logger.error("Model or processor is None")
            load_model()
            if model is None or processor is None:
                return "Error: Model not loaded properly"
        
        # Check if image file exists
        if not os.path.exists(image_path):
            return "Error: Image file not found"
        
        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Image loaded successfully: {image.size}")
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return f"Error loading image: {str(e)}"
        
        # Process image with error handling
        try:
            inputs = processor(image, return_tensors="pt")
            logger.info("Image processed by processor")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return f"Error processing image: {str(e)}"
        
        # Move inputs to same device as model
        try:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logger.info(f"Inputs moved to device: {device}")
        except Exception as e:
            logger.error(f"Error moving inputs to device: {str(e)}")
            return f"Error moving inputs to device: {str(e)}"
        
        # Generate caption
        try:
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50, num_beams=5)
            logger.info("Caption generated successfully")
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return f"Error during generation: {str(e)}"
        
        # Decode caption
        try:
            caption = processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"Caption decoded: {caption}")
            return caption if caption else "No caption generated"
        except Exception as e:
            logger.error(f"Error decoding caption: {str(e)}")
            return f"Error decoding caption: {str(e)}"
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_caption: {str(e)}")
        return f"Unexpected error: {str(e)}"


def image_to_base64(image_path):
    """Convert image to base64 for display in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Get image format
            with Image.open(image_path) as img:
                img_format = img.format.lower()
                
            return f"data:image/{img_format};base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return None


@app.route('/')
def index():
    """Main gallery page"""
    return render_template('gallery.html')


@app.route('/api/albums', methods=['GET'])
def get_albums():
    """Get all albums"""
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT * FROM albums ORDER BY updated_at DESC')
        albums = cursor.fetchall()
        db.close()
        
        albums_list = []
        for album in albums:
            cursor = get_db().cursor()
            cursor.execute('SELECT COUNT(*) as count FROM images WHERE album_id = ?', (album['id'],))
            image_count = cursor.fetchone()['count']
            get_db().close()
            
            albums_list.append({
                'id': album['id'],
                'name': album['name'],
                'description': album['description'],
                'image_count': image_count,
                'created_at': album['created_at'],
                'updated_at': album['updated_at']
            })
        
        return jsonify({'albums': albums_list})
    except Exception as e:
        logger.error(f"Error fetching albums: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/albums', methods=['POST'])
def create_album():
    """Create a new album"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        
        if not name:
            return jsonify({'error': 'Album name is required'}), 400
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute('INSERT INTO albums (name, description) VALUES (?, ?)',
                      (name, description))
        db.commit()
        album_id = cursor.lastrowid
        db.close()
        
        # Create album folder
        ensure_album_folder(album_id)
        
        return jsonify({'id': album_id, 'name': name, 'description': description, 'image_count': 0}), 201
    except Exception as e:
        logger.error(f"Error creating album: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/albums/<int:album_id>', methods=['DELETE'])
def delete_album(album_id):
    """Delete an album and all its images"""
    try:
        db = get_db()
        cursor = db.cursor()
        
        # Get all images in album
        cursor.execute('SELECT filename FROM images WHERE album_id = ?', (album_id,))
        images = cursor.fetchall()
        
        # Delete image files
        album_folder = get_album_folder(album_id)
        for image in images:
            filepath = os.path.join(album_folder, image['filename'])
            if os.path.exists(filepath):
                os.remove(filepath)
        
        # Delete from database
        cursor.execute('DELETE FROM images WHERE album_id = ?', (album_id,))
        cursor.execute('DELETE FROM albums WHERE id = ?', (album_id,))
        db.commit()
        db.close()
        
        # Remove album folder
        if os.path.exists(album_folder):
            os.rmdir(album_folder)
        
        return jsonify({'success': True}), 200
    except Exception as e:
        logger.error(f"Error deleting album: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/albums/<int:album_id>/images', methods=['GET'])
def get_album_images(album_id):
    """Get all images in an album"""
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT * FROM images WHERE album_id = ? ORDER BY uploaded_at DESC', (album_id,))
        images = cursor.fetchall()
        db.close()
        
        images_list = []
        for image in images:
            images_list.append({
                'id': image['id'],
                'filename': image['filename'],
                'original_filename': image['original_filename'],
                'caption': image['caption'],
                'uploaded_at': image['uploaded_at'],
                'album_id': album_id
            })
        
        return jsonify({'images': images_list})
    except Exception as e:
        logger.error(f"Error fetching album images: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/albums/<int:album_id>/upload', methods=['POST'])
def upload_images(album_id):
    """Upload images to an album"""
    try:
        # Validate album exists
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT id FROM albums WHERE id = ?', (album_id,))
        if not cursor.fetchone():
            return jsonify({'error': 'Album not found'}), 404
        db.close()
        
        # Ensure album folder exists
        album_folder = ensure_album_folder(album_id)
        
        # Handle multiple file uploads
        if 'files' not in request.files:
            return jsonify({'error': 'No files selected'}), 400
        
        files = request.files.getlist('files')
        uploaded_images = []
        
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                continue
            
            try:
                # Save file with timestamp
                original_filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                filename = timestamp + original_filename
                filepath = os.path.join(album_folder, filename)
                file.save(filepath)
                
                # Generate caption
                caption = generate_caption(filepath)
                
                # Save to database
                db = get_db()
                cursor = db.cursor()
                cursor.execute(
                    'INSERT INTO images (album_id, filename, original_filename, caption) VALUES (?, ?, ?, ?)',
                    (album_id, filename, original_filename, caption)
                )
                db.commit()
                image_id = cursor.lastrowid
                db.close()
                
                # Update album timestamp
                db = get_db()
                cursor = db.cursor()
                cursor.execute(
                    'UPDATE albums SET updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                    (album_id,)
                )
                db.commit()
                db.close()
                
                uploaded_images.append({
                    'id': image_id,
                    'filename': filename,
                    'original_filename': original_filename,
                    'caption': caption
                })
                logger.info(f"Image uploaded to album {album_id}: {filename}")
                
            except Exception as e:
                logger.error(f"Error uploading file {file.filename}: {str(e)}")
                continue
        
        if not uploaded_images:
            return jsonify({'error': 'No valid images were uploaded'}), 400
        
        return jsonify({'images': uploaded_images, 'success': True}), 201
    
    except Exception as e:
        logger.error(f"Error in upload_images: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/images/<int:image_id>/thumbnail')
def get_image_thumbnail(image_id):
    """Get image thumbnail as base64"""
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT album_id, filename FROM images WHERE id = ?', (image_id,))
        image = cursor.fetchone()
        db.close()
        
        if not image:
            return jsonify({'error': 'Image not found'}), 404
        
        filepath = os.path.join(get_album_folder(image['album_id']), image['filename'])
        base64_img = image_to_base64(filepath)
        
        if not base64_img:
            return jsonify({'error': 'Failed to process image'}), 500
        
        return jsonify({'image': base64_img})
    except Exception as e:
        logger.error(f"Error getting thumbnail: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/images/<int:image_id>/<field>')
def get_image_field(image_id, field):
    """Get specific image field (caption, etc.)"""
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute(f'SELECT {field} FROM images WHERE id = ?', (image_id,))
        result = cursor.fetchone()
        db.close()
        
        if not result:
            return jsonify({'error': 'Image not found'}), 404
        
        return jsonify({field: result[field]})
    except Exception as e:
        logger.error(f"Error getting image field: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/images/<int:image_id>/delete', methods=['POST'])
def delete_image(image_id):
    """Delete an image"""
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT album_id, filename FROM images WHERE id = ?', (image_id,))
        image = cursor.fetchone()
        
        if not image:
            return jsonify({'error': 'Image not found'}), 404
        
        # Delete file
        filepath = os.path.join(get_album_folder(image['album_id']), image['filename'])
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Delete from database
        cursor.execute('DELETE FROM images WHERE id = ?', (image_id,))
        db.commit()
        db.close()
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error deleting image: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'processor_loaded': processor is not None,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    })


gallery_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Personal Image Gallery</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .navbar {
            background: rgba(255, 255, 255, 0.95);
            padding: 1rem 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .navbar h1 {
            color: #667eea;
            font-size: 1.8em;
        }
        
        .nav-buttons {
            display: flex;
            gap: 10px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .view-section {
            display: none;
        }
        
        .view-section.active {
            display: block;
        }
        
        /* Albums View */
        .albums-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 2rem;
        }
        
        .album-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
        }
        
        .album-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        }
        
        .album-cover {
            height: 200px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 3em;
        }
        
        .album-info {
            padding: 20px;
        }
        
        .album-name {
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }
        
        .album-desc {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .album-meta {
            font-size: 0.85em;
            color: #999;
            border-top: 1px solid #eee;
            padding-top: 10px;
            margin-top: 10px;
        }
        
        .album-actions {
            display: flex;
            gap: 5px;
            margin-top: 10px;
        }
        
        .album-actions button {
            flex: 1;
            padding: 8px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85em;
            transition: background 0.3s;
        }
        
        .btn-open {
            background: #667eea;
            color: white;
        }
        
        .btn-open:hover {
            background: #5568d3;
        }
        
        .btn-delete {
            background: #ff6b6b;
            color: white;
        }
        
        .btn-delete:hover {
            background: #ee5a52;
        }
        
        /* Modal Styles */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        
        .modal.active {
            display: flex;
        }
        
        .modal-content {
            background: white;
            border-radius: 15px;
            padding: 30px;
            max-width: 500px;
            width: 90%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .modal-header {
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 20px;
            color: #333;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            color: #555;
            font-weight: 500;
        }
        
        .form-group input,
        .form-group textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-family: inherit;
            font-size: 1em;
        }
        
        .form-group textarea {
            resize: vertical;
            min-height: 80px;
        }
        
        .modal-buttons {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        .btn-primary {
            flex: 1;
            padding: 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
        }
        
        .btn-primary:hover {
            background: #5568d3;
        }
        
        .btn-secondary {
            flex: 1;
            padding: 12px;
            background: #f0f0f0;
            color: #333;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        
        /* Gallery View */
        .gallery-header {
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .back-btn {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
        }
        
        .back-btn:hover {
            background: #5568d3;
        }
        
        .upload-section {
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .upload-area:hover {
            border-color: #667eea;
            background: #f8fbff;
        }
        
        .upload-area.dragover {
            border-color: #667eea;
            background: #f0f8ff;
            transform: scale(1.02);
        }
        
        .file-input {
            display: none;
        }
        
        .images-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
        }
        
        .image-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
        }
        
        .image-container {
            width: 100%;
            height: 200px;
            overflow: hidden;
            background: #f0f0f0;
        }
        
        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .image-info {
            padding: 15px;
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .image-filename {
            font-weight: 500;
            color: #333;
            font-size: 0.9em;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            margin-bottom: 8px;
        }
        
        .image-caption {
            color: #666;
            font-size: 0.85em;
            line-height: 1.4;
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }
        
        .image-delete {
            background: #ff6b6b;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 6px;
            cursor: pointer;
            margin-top: 10px;
            font-size: 0.85em;
        }
        
        .image-delete:hover {
            background: #ee5a52;
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        
        .empty-state-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .empty-state-text {
            font-size: 1.2em;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            transition: background 0.3s;
        }
        
        .btn-success {
            background: #51cf66;
            color: white;
        }
        
        .btn-success:hover {
            background: #40c057;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .message {
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 15px;
            display: none;
        }
        
        .message.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .message.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .message.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>🖼️ Personal Image Gallery</h1>
        <div class="nav-buttons">
            <button class="btn btn-success" id="newAlbumBtn">+ New Album</button>
        </div>
    </div>
    
    <div class="container">
        <!-- Albums View -->
        <div id="albumsView" class="view-section active">
            <div id="albumsGrid" class="albums-grid"></div>
            <div id="emptyAlbums" class="empty-state" style="display: none;">
                <div class="empty-state-icon">📁</div>
                <div class="empty-state-text">No albums yet. Create your first album!</div>
            </div>
        </div>
        
        <!-- Gallery View -->
        <div id="galleryView" class="view-section">
            <div class="gallery-header">
                <button class="back-btn" id="backBtn">← Back to Albums</button>
                <h2 id="albumTitle"></h2>
                <div></div>
            </div>
            
            <div class="upload-section">
                <div class="message" id="uploadMessage"></div>
                <div class="upload-area" id="uploadArea">
                    <div>📸</div>
                    <div style="margin-top: 10px;">Drop images here or click to select</div>
                    <input type="file" id="fileInput" class="file-input" multiple accept="image/*">
                    <button type="button" class="btn btn-primary" style="margin-top: 10px;" onclick="document.getElementById('fileInput').click()">
                        Choose Images
                    </button>
                </div>
            </div>
            
            <div class="loading" id="uploadLoading">
                <div class="spinner"></div>
                <p>Uploading and generating captions...</p>
            </div>
            
            <div id="imagesGrid" class="images-grid"></div>
            <div id="emptyGallery" class="empty-state" style="display: none;">
                <div class="empty-state-icon">📷</div>
                <div class="empty-state-text">No images in this album yet. Upload some!</div>
            </div>
        </div>
    </div>
    
    <!-- New Album Modal -->
    <div id="newAlbumModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">Create New Album</div>
            <form id="newAlbumForm">
                <div class="form-group">
                    <label>Album Name *</label>
                    <input type="text" id="albumName" placeholder="e.g., Vacation 2025" required>
                </div>
                <div class="form-group">
                    <label>Description</label>
                    <textarea id="albumDesc" placeholder="Add a description (optional)"></textarea>
                </div>
                <div class="modal-buttons">
                    <button type="submit" class="btn-primary">Create Album</button>
                    <button type="button" class="btn-secondary" onclick="closeModal()">Cancel</button>
                </div>
            </form>
        </div>
    </div>
    
    <script>
        let currentAlbumId = null;
        
        // Toggle between views
        function showAlbumsView() {
            document.getElementById('albumsView').classList.add('active');
            document.getElementById('galleryView').classList.remove('active');
            loadAlbums();
        }
        
        function showGalleryView(albumId) {
            currentAlbumId = albumId;
            document.getElementById('albumsView').classList.remove('active');
            document.getElementById('galleryView').classList.add('active');
            loadAlbumImages(albumId);
        }
        
        // Initialize
        function init() {
            loadAlbums();
            setupEventListeners();
        }
        
        function setupEventListeners() {
            document.getElementById('newAlbumBtn').addEventListener('click', () => {
                document.getElementById('newAlbumModal').classList.add('active');
            });
            
            document.getElementById('backBtn').addEventListener('click', showAlbumsView);
            
            document.getElementById('newAlbumForm').addEventListener('submit', createAlbum);
            
            // Upload area
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    uploadFiles(files);
                }
            });
            
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    uploadFiles(e.target.files);
                }
            });
        }
        
        function closeModal() {
            document.getElementById('newAlbumModal').classList.remove('active');
            document.getElementById('newAlbumForm').reset();
        }
        
        function createAlbum(e) {
            e.preventDefault();
            
            const name = document.getElementById('albumName').value;
            const description = document.getElementById('albumDesc').value;
            
            fetch('/api/albums', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, description })
            })
            .then(r => r.json())
            .then(data => {
                if (data.id) {
                    closeModal();
                    loadAlbums();
                    showMessage('Album created successfully!', 'success');
                } else {
                    showMessage(data.error || 'Failed to create album', 'error');
                }
            })
            .catch(err => showMessage('Error creating album', 'error'));
        }
        
        function loadAlbums() {
            fetch('/api/albums')
                .then(r => r.json())
                .then(data => {
                    const grid = document.getElementById('albumsGrid');
                    const empty = document.getElementById('emptyAlbums');
                    
                    if (data.albums.length === 0) {
                        grid.innerHTML = '';
                        empty.style.display = 'block';
                    } else {
                        empty.style.display = 'none';
                        grid.innerHTML = data.albums.map(album => `
                            <div class="album-card">
                                <div class="album-cover">📁</div>
                                <div class="album-info">
                                    <div class="album-name">${album.name}</div>
                                    <div class="album-desc">${album.description || 'No description'}</div>
                                    <div class="album-meta">${album.image_count} images</div>
                                    <div class="album-actions">
                                        <button class="album-actions button btn-open" onclick="showGalleryView(${album.id})">Open</button>
                                        <button class="btn-delete" onclick="deleteAlbum(${album.id})">Delete</button>
                                    </div>
                                </div>
                            </div>
                        `).join('');
                    }
                })
                .catch(err => showMessage('Error loading albums', 'error'));
        }
        
        function deleteAlbum(albumId) {
            if (!confirm('Delete this album and all its images?')) return;
            
            fetch(`/api/albums/${albumId}`, { method: 'DELETE' })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        loadAlbums();
                        showMessage('Album deleted', 'success');
                    } else {
                        showMessage(data.error || 'Failed to delete', 'error');
                    }
                })
                .catch(err => showMessage('Error deleting album', 'error'));
        }
        
        function loadAlbumImages(albumId) {
            document.getElementById('albumTitle').textContent = '';
            
            fetch(`/api/albums/${albumId}/images`)
                .then(r => r.json())
                .then(data => {
                    const grid = document.getElementById('imagesGrid');
                    const empty = document.getElementById('emptyGallery');
                    
                    if (data.images.length === 0) {
                        grid.innerHTML = '';
                        empty.style.display = 'block';
                    } else {
                        empty.style.display = 'none';
                        grid.innerHTML = data.images.map(img => `
                            <div class="image-card">
                                <div class="image-container">
                                    <img src="" alt="Loading..." data-image-id="${img.id}">
                                </div>
                                <div class="image-info">
                                    <div class="image-filename">${img.original_filename}</div>
                                    <div class="image-caption">${img.caption || 'Generating caption...'}</div>
                                    <button class="image-delete" onclick="deleteImage(${img.id})">Delete</button>
                                </div>
                            </div>
                        `).join('');
                        
                        // Load images
                        data.images.forEach(img => {
                            fetch(`/api/images/${img.id}/thumbnail`)
                                .then(r => r.json())
                                .then(d => {
                                    const img_el = document.querySelector(`[data-image-id="${img.id}"]`);
                                    if (img_el) img_el.src = d.image;
                                });
                        });
                    }
                })
                .catch(err => showMessage('Error loading images', 'error'));
        }
        
        function uploadFiles(files) {
            if (!currentAlbumId) return;
            
            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
            }
            
            document.getElementById('uploadLoading').style.display = 'block';
            
            fetch(`/api/albums/${currentAlbumId}/upload`, {
                method: 'POST',
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('uploadLoading').style.display = 'none';
                if (data.success) {
                    loadAlbumImages(currentAlbumId);
                    document.getElementById('fileInput').value = '';
                    showMessage('Images uploaded successfully!', 'success');
                } else {
                    showMessage(data.error || 'Upload failed', 'error');
                }
            })
            .catch(err => {
                document.getElementById('uploadLoading').style.display = 'none';
                showMessage('Error uploading images', 'error');
            });
        }
        
        function deleteImage(imageId) {
            if (!confirm('Delete this image?')) return;
            
            fetch(`/api/images/${imageId}/delete`, { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        loadAlbumImages(currentAlbumId);
                        showMessage('Image deleted', 'success');
                    } else {
                        showMessage(data.error || 'Failed to delete', 'error');
                    }
                })
                .catch(err => showMessage('Error deleting image', 'error'));
        }
        
        function showMessage(text, type) {
            const msg = document.getElementById('uploadMessage');
            msg.textContent = text;
            msg.className = `message ${type} show`;
            setTimeout(() => msg.classList.remove('show'), 4000);
        }
        
        // Start
        init();
    </script>
</body>
</html>
'''

# Create templates directory and save template
templates_dir = 'templates'
os.makedirs(templates_dir, exist_ok=True)

# Write template
try:
    with open(os.path.join(templates_dir, 'gallery.html'), 'w', encoding='utf-8') as f:
        f.write(gallery_template)
except UnicodeEncodeError:
    clean_template = gallery_template.encode('ascii', 'ignore').decode('ascii')
    with open(os.path.join(templates_dir, 'gallery.html'), 'w', encoding='utf-8') as f:
        f.write(clean_template)

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Load model on startup
    try:
        load_model()
        print("🚀 Starting Flask application...")
        print("📝 Model loaded successfully!")
        print("🌐 Open http://localhost:8000 in your browser")
        app.run(debug=False, host='0.0.0.0', port=8000, use_reloader=False)
    except Exception as e:
        print(f"❌ Failed to start application: {str(e)}")
        print("Make sure you have the required dependencies installed:")
        print("pip install flask torch transformers pillow")
