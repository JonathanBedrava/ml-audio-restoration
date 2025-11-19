#!/usr/bin/env node

import { Command } from 'commander';
import * as dotenv from 'dotenv';
import * as path from 'path';
import * as fs from 'fs';
import PQueue from 'p-queue';
import { FreesoundScraper } from './scrapers/FreesoundScraper';
import { InternetArchiveScraper } from './scrapers/InternetArchiveScraper';
import { MusopenScraper } from './scrapers/MusopenScraper';
import { BaseScraper } from './scrapers/BaseScraper';
import { ScraperConfig } from './types';

// Load environment variables
dotenv.config();

const program = new Command();

program
  .name('audio-data-scraper')
  .description('Download stereo audio training data from various sources')
  .version('1.0.0')
  .option('-s, --source <source>', 'Source to scrape (freesound|archive|musopen|all)', 'all')
  .option('-o, --output <dir>', 'Output directory', process.env.OUTPUT_DIR || '../data/raw')
  .option('-m, --max-files <number>', 'Maximum files per source', process.env.MAX_FILES_PER_SOURCE || '50')
  .option('-c, --concurrent <number>', 'Max concurrent downloads', process.env.MAX_CONCURRENT_DOWNLOADS || '3')
  .option('--min-duration <seconds>', 'Minimum audio duration', process.env.MIN_DURATION_SECONDS || '30')
  .option('--max-duration <seconds>', 'Maximum audio duration', process.env.MAX_DURATION_SECONDS || '600');

program.parse();

const options = program.opts();

async function main() {
  console.log('='.repeat(60));
  console.log('Audio Data Scraper for ML Training');
  console.log('='.repeat(60));
  console.log(`Source: ${options.source}`);
  console.log(`Output: ${options.output}`);
  console.log(`Max files per source: ${options.maxFiles}`);
  console.log(`Concurrent downloads: ${options.concurrent}`);
  console.log('='.repeat(60));

  // Ensure output directory exists
  const outputDir = path.resolve(__dirname, '..', options.output);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
    console.log(`Created output directory: ${outputDir}`);
  }

  // Configuration
  const config: ScraperConfig = {
    apiKey: process.env.FREESOUND_API_KEY,
    outputDir,
    maxConcurrentDownloads: parseInt(options.concurrent),
    maxFilesPerSource: parseInt(options.maxFiles),
    minDuration: parseInt(options.minDuration),
    maxDuration: parseInt(options.maxDuration),
    genres: ['jazz', 'classical'],
    requiredSampleRate: 44100,
    requiredChannels: 2
  };

  // Create scrapers based on source
  const scrapers: BaseScraper[] = [];

  if (options.source === 'all' || options.source === 'freesound') {
    if (!config.apiKey) {
      console.warn('\n⚠ Warning: FREESOUND_API_KEY not found in .env file');
      console.warn('Get an API key at: https://freesound.org/apiv2/apply/');
      console.warn('Skipping Freesound...\n');
    } else {
      scrapers.push(new FreesoundScraper(config));
    }
  }

  if (options.source === 'all' || options.source === 'archive') {
    scrapers.push(new InternetArchiveScraper(config));
  }

  if (options.source === 'all' || options.source === 'musopen') {
    scrapers.push(new MusopenScraper(config));
  }

  if (scrapers.length === 0) {
    console.error('No valid scrapers configured. Check your settings and API keys.');
    process.exit(1);
  }

  // Run scrapers
  for (const scraper of scrapers) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Starting ${scraper.getName()} scraper...`);
    console.log('='.repeat(60));

    try {
      // Search for files
      const files = await scraper.search();
      console.log(`\nFound ${files.length} eligible files from ${scraper.getName()}`);

      if (files.length === 0) {
        console.log('No files to download');
        continue;
      }

      // Download with concurrency control
      console.log(`\nStarting downloads (max ${config.maxConcurrentDownloads} concurrent)...`);
      const queue = new PQueue({ concurrency: config.maxConcurrentDownloads });

      const downloadPromises = files.map(file => 
        queue.add(() => scraper.download(file))
      );

      await Promise.all(downloadPromises);

      // Print statistics
      scraper.printStats();

    } catch (error: any) {
      console.error(`Error running ${scraper.getName()} scraper: ${error.message}`);
    }
  }

  console.log('\n' + '='.repeat(60));
  console.log('Scraping complete!');
  console.log(`Check your files in: ${outputDir}`);
  console.log('='.repeat(60) + '\n');
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
