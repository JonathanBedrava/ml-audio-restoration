import { BaseScraper } from './BaseScraper';
import { AudioFile, ScraperConfig } from '../types';

export class MusopenScraper extends BaseScraper {
  private readonly BASE_URL = 'https://api.musopen.org/music';

  getName(): string {
    return 'Musopen';
  }

  async search(): Promise<AudioFile[]> {
    console.log('\nSearching Musopen...');
    console.log('Note: Musopen API requires authentication for downloads.');
    console.log('This scraper will provide direct links, but you may need to:');
    console.log('1. Create a free account at https://musopen.org');
    console.log('2. Browse and download manually, or');
    console.log('3. Contact Musopen for API access\n');

    const eligibleFiles: AudioFile[] = [];

    try {
      // Public API endpoint (limited)
      const response = await this.client.get(this.BASE_URL, {
        params: {
          format: 'json',
          limit: 100
        }
      });

      const recordings = response.data.results || [];
      console.log(`Found ${recordings.length} recordings`);

      for (const recording of recordings) {
        // Musopen recordings info
        const file: AudioFile = {
          id: recording.id?.toString() || 'unknown',
          name: recording.title || 'Unknown',
          url: `https://musopen.org/music/${recording.id}/`,
          downloadUrl: `https://musopen.org/music/${recording.id}/`, // User will need to download manually
          duration: 180, // Approximate, since not provided by API
          sampleRate: 44100, // Assume standard quality
          channels: 2, // Assume stereo
          format: 'flac',
          license: 'Public Domain',
          source: 'musopen'
        };

        console.log(`Found: ${file.name}`);
        console.log(`  URL: ${file.url}`);
        eligibleFiles.push(file);
        this.stats.eligible++;

        if (eligibleFiles.length >= this.config.maxFilesPerSource) {
          break;
        }
      }

    } catch (error: any) {
      console.error(`Error accessing Musopen API: ${error.message}`);
      console.log('\nAlternative: Visit https://musopen.org/music/ to browse and download manually.');
      console.log('Musopen offers free public domain classical music recordings.');
    }

    this.stats.searched = eligibleFiles.length;
    return eligibleFiles;
  }

  async download(file: AudioFile): Promise<boolean> {
    console.log(`\nMusopen download requires manual action:`);
    console.log(`1. Visit: ${file.url}`);
    console.log(`2. Click the download button`);
    console.log(`3. Save to: ${this.config.outputDir}`);
    console.log(`4. Rename to include "musopen_${file.id}_" prefix`);
    
    this.stats.skipped++;
    return false;
  }
}
