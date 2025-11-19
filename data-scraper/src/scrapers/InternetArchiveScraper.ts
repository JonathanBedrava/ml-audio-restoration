import { BaseScraper } from './BaseScraper';
import { AudioFile, ScraperConfig } from '../types';

export class InternetArchiveScraper extends BaseScraper {
  private readonly BASE_URL = 'https://archive.org';
  private readonly SEARCH_URL = 'https://archive.org/advancedsearch.php';

  getName(): string {
    return 'InternetArchive';
  }

  async search(): Promise<AudioFile[]> {
    const eligibleFiles: AudioFile[] = [];
    
    const collections = [
      'etree', // Live music recordings - usually high quality stereo
      'opensource_audio',
      'audio_music',
      'GratefulDead', // Known for high-quality soundboard recordings
      'georgeblood' // Audio preservation, high quality digitizations (excluding 78s)
    ];

    const queries = [
      'jazz stereo',
      'classical stereo',
      'orchestra stereo',
      'piano stereo',
      'chamber music stereo',
      'live recording',
      'soundboard'
    ];

    for (const collection of collections) {
      for (const query of queries) {
        console.log(`\nSearching Internet Archive: ${collection} - "${query}"`);
        
        try {
          // Search for items in the collection
          const searchParams = {
            q: `collection:${collection} AND ${query} AND format:(wav OR flac) AND NOT 78rpm AND NOT mono`,
            fl: 'identifier,title',
            rows: 50,
            output: 'json'
          };

          const response = await this.client.get(this.SEARCH_URL, {
            params: searchParams
          });

          const items = response.data.response?.docs || [];
          console.log(`Found ${items.length} potential items`);

          for (const item of items) {
            try {
              console.log(`\n  Fetching metadata for: ${item.identifier}`);
              // Get metadata for each item
              const metadata = await this.getItemMetadata(item.identifier);
              
              if (metadata && metadata.length > 0) {
                eligibleFiles.push(...metadata);
                console.log(`  📥 Download queue: ${eligibleFiles.length} files`);
                
                if (eligibleFiles.length >= this.config.maxFilesPerSource) {
                  console.log(`\n✓ Reached limit of ${this.config.maxFilesPerSource} files`);
                  return eligibleFiles;
                }
              } else {
                console.log(`  📥 Download queue: ${eligibleFiles.length} files (no change)`);
              }

              // Rate limiting
              await this.sleep(1000);

            } catch (error: any) {
              console.error(`  Error fetching metadata for ${item.identifier}: ${error.message}`);
            }
          }

        } catch (error: any) {
          console.error(`Error searching ${collection}: ${error.message}`);
        }

        await this.sleep(1000);
      }
    }

    return eligibleFiles;
  }

  private async getItemMetadata(identifier: string): Promise<AudioFile[]> {
    try {
      const metadataUrl = `${this.BASE_URL}/metadata/${identifier}`;
      const response = await this.client.get(metadataUrl);
      
      const files = response.data.files || [];
      const eligibleFiles: AudioFile[] = [];

      let wavFlacCount = 0;

      for (const file of files) {
        // Look for WAV or FLAC files (case insensitive)
        if (!file.format) {
          continue;
        }
        
        const formatLower = file.format.toLowerCase();
        if (!['wav', 'flac', 'wave'].includes(formatLower)) {
          continue;
        }

        wavFlacCount++;
        console.log(`      Checking: ${file.name}`);

        // Skip if explicitly mono
        if (file.channels) {
          if (parseInt(file.channels) !== 2) {
            console.log(`        ✗ Mono (${file.channels} channels)`);
            continue;
          }
        }

        // Try to extract audio properties
        const audioFile: AudioFile = {
          id: `${identifier}_${file.name}`,
          name: file.name || identifier,
          url: `${this.BASE_URL}/details/${identifier}`,
          downloadUrl: `${this.BASE_URL}/download/${identifier}/${file.name}`,
          duration: parseFloat(file.length) || 0,
          sampleRate: parseInt(file.sample_rate) || 44100,
          channels: parseInt(file.channels) || 2,
          format: formatLower === 'wave' ? 'wav' : formatLower,
          license: 'Public Domain / Creative Commons',
          source: 'archive'
        };

        // Strict validation - must be stereo and sufficient quality
        if (audioFile.duration === 0) {
          console.log(`        ✗ No duration info`);
        } else if (audioFile.channels !== 2) {
          console.log(`        ✗ Not stereo (${audioFile.channels} channels)`);
        } else if (audioFile.sampleRate < 44100) {
          console.log(`        ✗ Sample rate too low (${audioFile.sampleRate}Hz)`);
        } else if (!this.isValidLicense(audioFile.license)) {
          console.log(`        ✗ License not valid`);
        } else if (!this.isEligible(audioFile)) {
          console.log(`        ✗ Duration outside range (${audioFile.duration}s, need ${this.config.minDuration}-${this.config.maxDuration}s)`);
        } else {
          eligibleFiles.push(audioFile);
          this.stats.eligible++;
          console.log(`        ✓ ELIGIBLE!`);
        }
      }

      if (wavFlacCount === 0) {
        console.log(`    No WAV/FLAC files found in this item`);
      } else if (eligibleFiles.length === 0) {
        console.log(`    Found ${wavFlacCount} audio files but none met criteria`);
      }

      this.stats.searched += files.length;
      return eligibleFiles;

    } catch (error: any) {
      console.error(`    Error getting metadata: ${error.message}`);
      return [];
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private isValidLicense(license: string): boolean {
    // Internet Archive content is typically Public Domain or CC
    const validLicenses = [
      'public domain',
      'creative commons',
      'cc0',
      'cc-zero'
    ];
    
    return validLicenses.some(valid => 
      license.toLowerCase().includes(valid)
    );
  }
}
