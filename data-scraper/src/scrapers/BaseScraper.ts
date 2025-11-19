import axios, { AxiosInstance } from 'axios';
import * as fs from 'fs';
import * as path from 'path';
import { AudioFile, ScraperConfig, ScraperStats } from '../types';

export abstract class BaseScraper {
  protected config: ScraperConfig;
  protected stats: ScraperStats;
  protected client: AxiosInstance;

  constructor(config: ScraperConfig) {
    this.config = config;
    this.stats = {
      searched: 0,
      eligible: 0,
      downloaded: 0,
      failed: 0,
      skipped: 0
    };
    
    this.client = axios.create({
      timeout: 30000,
      headers: {
        'User-Agent': 'AudioDataScraper/1.0'
      }
    });
  }

  abstract search(): Promise<AudioFile[]>;
  abstract getName(): string;

  protected isEligible(file: AudioFile): boolean {
    // Check sample rate (44100Hz or higher)
    if (file.sampleRate < this.config.requiredSampleRate) {
      return false;
    }

    // Check stereo (2 channels)
    if (file.channels !== this.config.requiredChannels) {
      return false;
    }

    // Check duration
    if (file.duration < this.config.minDuration || file.duration > this.config.maxDuration) {
      return false;
    }

    // Check format (WAV or FLAC)
    const allowedFormats = ['wav', 'flac'];
    if (!allowedFormats.includes(file.format.toLowerCase())) {
      return false;
    }

    return true;
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

      const response = await this.client.get(file.downloadUrl, {
        responseType: 'stream',
        timeout: 300000 // 5 minutes for large files
      });

      const writer = fs.createWriteStream(filePath);
      response.data.pipe(writer);

      return new Promise((resolve, reject) => {
        writer.on('finish', () => {
          this.stats.downloaded++;
          console.log(`✓ Downloaded: ${fileName}`);
          resolve(true);
        });
        writer.on('error', (err) => {
          this.stats.failed++;
          console.error(`✗ Failed: ${fileName} - ${err.message}`);
          reject(err);
        });
      });
    } catch (error) {
      this.stats.failed++;
      console.error(`✗ Download failed: ${file.name} - ${error}`);
      return false;
    }
  }

  protected sanitizeFilename(filename: string): string {
    // Remove or replace invalid characters
    return filename
      .replace(/[<>:"|?*]/g, '')
      .replace(/\s+/g, '_')
      .substring(0, 100); // Limit length
  }

  getStats(): ScraperStats {
    return this.stats;
  }

  printStats(): void {
    console.log('\n' + '='.repeat(50));
    console.log(`${this.getName()} Statistics:`);
    console.log('='.repeat(50));
    console.log(`Searched:    ${this.stats.searched}`);
    console.log(`Eligible:    ${this.stats.eligible}`);
    console.log(`Downloaded:  ${this.stats.downloaded}`);
    console.log(`Skipped:     ${this.stats.skipped}`);
    console.log(`Failed:      ${this.stats.failed}`);
    console.log('='.repeat(50) + '\n');
  }
}
