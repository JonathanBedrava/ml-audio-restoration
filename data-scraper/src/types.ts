export interface AudioFile {
  id: string;
  name: string;
  url: string;
  downloadUrl: string;
  duration: number;
  sampleRate: number;
  channels: number;
  format: string;
  license: string;
  source: 'freesound' | 'archive' | 'musopen';
}

export interface ScraperConfig {
  apiKey?: string;
  outputDir: string;
  maxConcurrentDownloads: number;
  maxFilesPerSource: number;
  minDuration: number;
  maxDuration: number;
  genres: string[];
  requiredSampleRate: number;
  requiredChannels: number;
}

export interface ScraperStats {
  searched: number;
  eligible: number;
  downloaded: number;
  failed: number;
  skipped: number;
}
