# SigLIP API Server

## Quick Start

### 1. Start server
```bash
docker-compose up -d
```

### 2. Check logs
```bash
docker-compose logs -f
```

### 3. Test API
```bash
python test_api.py
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Classify
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": ["Clothing", "Shoes", "Bag"],
    "text": "red dress",
    "image_url": "https://example.com/image.jpg"
  }'
```

## Performance

- Cold start: ~60s (first time)
- Warm inference: ~2-3s
- Model always in memory

## Integration with Laravel

Add to your docker-compose.yml:
```yaml
services:
  laravel:
    # ... your Laravel config
    depends_on:
      - siglip-api
    environment:
      - SIGLIP_API_URL=http://siglip-api:8000
```

Use in PHP:
```php
$response = Http::post('http://siglip-api:8000/classify', [
    'keywords' => ['Clothing', 'Shoes'],
    'image_url' => $imageUrl
]);
```
