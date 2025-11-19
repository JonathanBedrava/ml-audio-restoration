# Audio Data Scraper

TypeScript-based scraper for downloading high-quality stereo audio training data from Freesound, Internet Archive, and Musopen.

## Features

- **Multiple Sources**: Freesound, Internet Archive, and Musopen
- **Smart Filtering**: Automatically filters for stereo (2 channel), 44.1kHz+ WAV/FLAC files
- **Genre Targeting**: Focuses on jazz and classical music
- **Concurrent Downloads**: Configurable parallel downloading
- **Rate Limiting**: Respects API limits

## Setup

1. **Install dependencies:**
```powershell
cd data-scraper
npm install
```

2. **Configure environment:**
```powershell
cp .env.example .env
```

3. **Edit `.env` file:**
   - **Freesound**: Get API key at https://freesound.org/apiv2/apply/ (free, instant approval)
   - **Internet Archive**: No key needed (optional: add your email)
   - **Musopen**: Manual downloads (links will be provided)

## Usage

**Scrape from all sources:**
```powershell
npm run scrape:all
```

**Scrape from specific source:**
```powershell
npm run scrape:freesound
npm run scrape:archive
npm run scrape:musopen
```

**Custom options:**
```powershell
npm run dev -- --source freesound --max-files 100 --concurrent 5
```

### Options

- `--source <source>`: Source to scrape (freesound|archive|musopen|all)
- `--output <dir>`: Output directory (default: `../data/raw`)
- `--max-files <number>`: Maximum files per source (default: 50)
- `--concurrent <number>`: Max concurrent downloads (default: 3)
- `--min-duration <seconds>`: Minimum audio duration (default: 30)
- `--max-duration <seconds>`: Maximum audio duration (default: 600)

## What Gets Downloaded

- **Format**: WAV or FLAC only (lossless)
- **Sample Rate**: 44.1kHz or higher
- **Channels**: Stereo (2 channels)
- **Duration**: 30 seconds to 10 minutes
- **Genres**: Jazz and classical music
- **License**: Creative Commons or Public Domain

## Source Details

### Freesound
- Requires free API key
- Large community audio library
- Good for jazz ensemble recordings, live music
- Downloads automatically

### Internet Archive
- No API key required
- Excellent for classical, orchestral, and 78rpm collections
- Public domain content
- Downloads automatically

### Musopen
- Public domain classical music
- High-quality recordings
- API access limited - provides URLs for manual download
- Visit provided links to download

## Output

Files are saved to `../data/raw/` with the format:
```
{source}_{id}_{original_filename}
```

Example:
```
freesound_123456_jazz_piano_solo.wav
archive_78rpm_symphony_no5.flac
```

## Tips

1. **Start with Freesound** - Easiest to set up and most reliable
2. **Internet Archive** - Best for variety, but slower
3. **Musopen** - Manual downloads, but highest quality classical
4. **Be patient** - High-quality audio files are large (100MB+)
5. **Check licenses** - All sources provide proper attribution info

## Troubleshooting

**"Freesound API key required"**
- Get a free key at https://freesound.org/apiv2/apply/
- Add to `.env` file as `FREESOUND_API_KEY=your_key_here`

**"No files found"**
- Sources may have limited stereo 44.1kHz+ content
- Try increasing `--max-duration` or reducing `--min-duration`
- Some genres have more stereo content than others

**"Download failed"**
- Some files may be unavailable or moved
- Increase `--concurrent` value if downloads are slow
- Decrease if you're getting rate limited

## Building

```powershell
npm run build
npm start -- --source freesound
```

## Notes

- Downloaded files will be used for training ML audio restoration models
- All sources provide properly licensed content for ML training
- Total download size can be 5-10GB+ depending on settings
