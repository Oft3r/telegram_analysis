# Telegram Channel Analyzer

An AI-powered Telegram channel analyzer that provides detailed insights about channel content and engagement using Google's Gemini AI.

## Features

- Message analysis using Google's Gemini AI
- Media analysis (photos and videos)
- Link, hashtag, and mention detection
- Engagement metrics calculation
- Detailed report generation
- Automatic channel rating system
- Peak activity hours detection

## Technical Requirements

- Python 3.8 or higher
- Telegram API credentials
- Google Gemini API key

## Dependencies

```
telethon==1.34.0
google-generativeai==0.3.2
pandas==2.2.0
python-dotenv==1.0.1
asyncio==3.4.3
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/telegram-analyzer.git
cd telegram-analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your credentials:
```env
API_ID=your_api_id
API_HASH=your_api_hash
GEMINI_API_KEY=your_gemini_api_key
```

## How to Get API Keys

### Telegram API Credentials
1. Visit https://my.telegram.org/auth
2. Log in with your phone number
3. Go to 'API development tools'
4. Create a new application
5. Copy the API_ID and API_HASH

### Google Gemini API Key
1. Visit https://makersuite.google.com/app/apikey
2. Create or select a project
3. Generate an API key

## Usage

Run the analyzer:
```bash
python telegram_analyzer.py
```

When prompted, enter the Telegram channel URL you want to analyze (format: https://t.me/channelname).

## Output

The analyzer generates:
1. Console output with key metrics
2. A detailed markdown report (`telegram_analysis_report.md`) containing:
   - General channel analysis
   - Top messages by engagement
   - Most shared links
   - Popular hashtags
   - Frequent mentions
   - Channel ratings

## Analysis Metrics

The tool provides various metrics including:
- Message frequency and timing
- Engagement rates (views, forwards, replies)
- Media usage patterns
- Link sharing analysis
- Hashtag popularity
- Channel ratings for:
  - Activity level
  - Content quality
  - Community engagement
  - Trustworthiness

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for educational and analytical purposes only. Please respect Telegram's terms of service and API usage guidelines when using this analyzer.

## Author

@Oft3r

## Acknowledgments

- Telethon library for Telegram API access
- Google's Gemini AI for content analysis
- Pandas for data processing
