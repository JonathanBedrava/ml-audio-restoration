import { BaseScraper } from './BaseScraper';
import { AudioFile, ScraperConfig } from '../types';
import * as path from 'path';
import * as fs from 'fs';
import axios from 'axios';
import { FreesoundOAuth } from '../auth/FreesoundOAuth';

export class FreesoundScraper extends BaseScraper {
  private readonly BASE_URL = 'https://freesound.org/apiv2';
  private oauth: FreesoundOAuth | null = null;

  constructor(config: ScraperConfig) {
    super(config);
    if (!config.apiKey) {
      throw new Error('Freesound API key is required. Get one at https://freesound.org/apiv2/apply/');
    }

    // Initialize OAuth if we have client credentials from environment
    const clientId = process.env.FREESOUND_CLIENT_ID;
    const clientSecret = process.env.FREESOUND_CLIENT_SECRET;

    if (clientId && clientSecret) {
      this.oauth = new FreesoundOAuth(clientId, clientSecret, './config');
      console.log('✅ Freesound OAuth initialized');
    } else {
      console.log('⚠️  FREESOUND_CLIENT_ID or FREESOUND_CLIENT_SECRET not found');
      console.log('⚠️  Downloads will not be available without OAuth');
    }
  }

  getName(): string {
    return 'Freesound';
  }

  async search(): Promise<AudioFile[]> {
    const eligibleFiles: AudioFile[] = [];
    const queries = [
      'jazz quartet',
      'jazz quintet',
      'classical piano',
      'classical orchestra',
      'string quartet',
      'chamber music ',
      'symphony',
      'solo violin',
      'opera',
      'live jazz recording',
      'concert recording'
    ];

    for (const query of queries) {
      console.log(`\nSearching Freesound for: "${query}"`);
      
      try {
        const response = await this.client.get(`${this.BASE_URL}/search/text/`, {
          params: {
            query: query,
            filter: 'channels:2 samplerate:44100 (type:wav OR type:flac) (license:"Creative Commons 0" OR license:"Attribution")',
            fields: 'id,name,duration,channels,samplerate,type,download,previews,license',
            page_size: 150,
            token: this.config.apiKey
          }
        });

        const results = response.data.results || [];
        this.stats.searched += results.length;

        for (const item of results) {
          const file: AudioFile = {
            id: item.id.toString(),
            name: item.name,
            url: item.url,
            downloadUrl: item.download,
            duration: item.duration,
            sampleRate: item.samplerate,
            channels: item.channels,
            format: item.type,
            license: item.license,
            source: 'freesound'
          };

          console.log(`  Checking: ${file.name}`);
          console.log(`    Duration: ${file.duration}s, Sample Rate: ${file.sampleRate}Hz, Channels: ${file.channels}, License: ${file.license}`);

          // Ensure it's actually stereo and meets requirements
          if (file.channels !== 2) {
            console.log(`    ✗ Skipped: not stereo (${file.channels} channels)`);
          } else if (file.sampleRate < 44100) {
            console.log(`    ✗ Skipped: sample rate too low (${file.sampleRate}Hz)`);
          } else if (!this.isValidLicense(file.license)) {
            console.log(`    ✗ Skipped: license not valid (${file.license})`);
          } else if (!this.isEligible(file)) {
            console.log(`    ✗ Skipped: failed eligibility check (duration or format)`);
          } else {
            eligibleFiles.push(file);
            this.stats.eligible++;
            console.log(`    ✓ ELIGIBLE! Adding to download queue`);
            console.log(`  📥 Download queue: ${eligibleFiles.length} files`);
            
            if (eligibleFiles.length >= this.config.maxFilesPerSource) {
              console.log(`\n✓ Reached limit of ${this.config.maxFilesPerSource} files`);
              return eligibleFiles;
            }
          }
        }

        // Rate limiting - slow down to avoid 503 errors
        await this.sleep(2000);

      } catch (error: any) {
        console.error(`Error searching "${query}": ${error.message}`);
        // Back off more if we get rate limited
        if (error.response?.status === 503) {
          console.log('Rate limited, waiting 10 seconds...');
          await this.sleep(10000);
        }
      }
    }

    return eligibleFiles;
  }

  async download(file: AudioFile): Promise<boolean> {
    try {
      const fileName = this.sanitizeFilename(file.name);
      const filePath = path.join(
        this.config.outputDir,
        `${this.getName()}_${file.id}_${fileName}`
      );

      // Check if already downloaded
      if (fs.existsSync(filePath)) {
        console.log(`Skipping (already exists): ${fileName}`);
        this.stats.skipped++;
        return false;
      }

      console.log(`Downloading: ${fileName} (${file.duration}s, ${file.sampleRate}Hz, ${file.channels}ch)`);

      if (!this.oauth) {
        console.log(`  ⚠️  OAuth not configured - cannot download`);
        console.log(`  ℹ️  Manual download: https://freesound.org/s/sounds/${file.id}/`);
        this.stats.skipped++;
        return false;
      }

      // Use OAuth to download
      await this.oauth.downloadFile(file.downloadUrl, filePath);
      this.stats.downloaded++;
      console.log(`✓ Downloaded: ${fileName}`);
      return true;

    } catch (error: any) {
      this.stats.failed++;
      console.error(`✗ Download failed: ${file.name} - ${error.message}`);
      return false;
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private isValidLicense(license: string): boolean {
    // Accept CC0, Public Domain, and CC-BY (attribution only, no SA/NC restrictions)
    const validPatterns = [
      'publicdomain/zero',
      'public domain',
      'cc0',
      '/by/3.0',
      '/by/4.0',
      'creative commons attribution'
    ];
    
    const lower = license.toLowerCase();
    
    // Reject NC (non-commercial) and ND (no derivatives) - those restrict ML training
    if (lower.includes('/by-nc') || lower.includes('/by-nd') || lower.includes('/by-sa')) {
      return false;
    }
    
    return validPatterns.some(pattern => lower.includes(pattern));
  }
}
