import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';
import { createServer } from 'http';
import { parse } from 'url';
import open from 'open';

interface TokenData {
  access_token: string;
  refresh_token: string;
  expires_in: number;
  expires_at: number;
  token_type: string;
  created_at: string;
}

export class FreesoundOAuth {
  private clientId: string;
  private clientSecret: string;
  private redirectUri: string = 'http://localhost:3000/callback';
  private baseUrl: string = 'https://freesound.org';
  private tokenPath: string;
  private cachedToken: string | null = null;
  private tokenCacheExpiry: number | null = null;
  private authInProgress: Promise<TokenData> | null = null;

  constructor(clientId: string, clientSecret: string, configDir: string = './config') {
    this.clientId = clientId;
    this.clientSecret = clientSecret;
    this.tokenPath = path.join(configDir, 'freesound-token.json');

    if (!this.clientId || !this.clientSecret) {
      throw new Error('FREESOUND_CLIENT_ID and FREESOUND_CLIENT_SECRET are required for OAuth');
    }

    // Ensure config directory exists
    const dir = path.dirname(this.tokenPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  /**
   * Start the OAuth authorization flow
   */
  async authorize(): Promise<TokenData> {
    return new Promise((resolve, reject) => {
      console.log('🔐 Starting Freesound OAuth authorization...');

      const server = createServer((req, res) => {
        const parsedUrl = parse(req.url || '', true);

        if (parsedUrl.pathname === '/callback') {
          const code = parsedUrl.query.code as string;
          const error = parsedUrl.query.error as string;

          if (error) {
            res.writeHead(400, { 'Content-Type': 'text/html' });
            res.end('<h1>Authorization Failed</h1><p>Error: ' + error + '</p>');
            server.close();
            reject(new Error('Authorization failed: ' + error));
            return;
          }

          if (code) {
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end('<h1>Authorization Successful!</h1><p>You can close this window and return to the terminal.</p>');
            server.close();

            this.exchangeCodeForToken(code)
              .then(resolve)
              .catch(reject);
          } else {
            res.writeHead(400, { 'Content-Type': 'text/html' });
            res.end('<h1>Authorization Failed</h1><p>No authorization code received.</p>');
            server.close();
            reject(new Error('No authorization code received'));
          }
        } else {
          res.writeHead(404, { 'Content-Type': 'text/html' });
          res.end('<h1>Not Found</h1>');
        }
      });

      server.listen(3000, () => {
        const authUrl = `${this.baseUrl}/apiv2/oauth2/authorize/?` +
          `client_id=${this.clientId}&` +
          `response_type=code&` +
          `state=xyz&` +
          `redirect_uri=${encodeURIComponent(this.redirectUri)}`;

        console.log('🌐 Opening authorization URL in browser...');
        console.log('📋 If the browser doesn\'t open automatically, visit:');
        console.log(`   ${authUrl}`);

        open(authUrl).catch(() => {
          console.log('⚠️  Could not open browser automatically. Please visit the URL above manually.');
        });
      });

      // Timeout after 5 minutes
      setTimeout(() => {
        server.close();
        reject(new Error('Authorization timeout - please try again'));
      }, 300000);
    });
  }

  /**
   * Exchange authorization code for access token
   */
  private async exchangeCodeForToken(code: string): Promise<TokenData> {
    try {
      console.log('🔄 Exchanging authorization code for access token...');

      const params = new URLSearchParams();
      params.append('client_id', this.clientId);
      params.append('client_secret', this.clientSecret);
      params.append('grant_type', 'authorization_code');
      params.append('code', code);
      params.append('redirect_uri', this.redirectUri);

      const response = await axios.post(`${this.baseUrl}/apiv2/oauth2/access_token/`, params, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });

      const tokenData: TokenData = {
        access_token: response.data.access_token,
        refresh_token: response.data.refresh_token,
        expires_in: response.data.expires_in,
        expires_at: Date.now() + (response.data.expires_in * 1000),
        token_type: response.data.token_type || 'Bearer',
        created_at: new Date().toISOString()
      };

      fs.writeFileSync(this.tokenPath, JSON.stringify(tokenData, null, 2));

      this.cachedToken = tokenData.access_token;
      this.tokenCacheExpiry = tokenData.expires_at - 300000;

      console.log('✅ OAuth token saved successfully');
      console.log(`🕐 Token expires: ${new Date(tokenData.expires_at).toLocaleString()}`);

      return tokenData;
    } catch (error: any) {
      console.error('❌ Failed to exchange code for token:', error.response?.data || error.message);
      throw error;
    }
  }

  /**
   * Refresh expired token
   */
  private async refreshToken(tokenData: TokenData): Promise<TokenData | null> {
    try {
      console.log('🔄 Refreshing expired token...');

      const params = new URLSearchParams();
      params.append('client_id', this.clientId);
      params.append('client_secret', this.clientSecret);
      params.append('grant_type', 'refresh_token');
      params.append('refresh_token', tokenData.refresh_token);

      const response = await axios.post(`${this.baseUrl}/apiv2/oauth2/access_token/`, params, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });

      const newTokenData: TokenData = {
        access_token: response.data.access_token,
        refresh_token: response.data.refresh_token || tokenData.refresh_token,
        expires_in: response.data.expires_in,
        expires_at: Date.now() + (response.data.expires_in * 1000),
        token_type: response.data.token_type || 'Bearer',
        created_at: new Date().toISOString()
      };

      fs.writeFileSync(this.tokenPath, JSON.stringify(newTokenData, null, 2));

      this.cachedToken = newTokenData.access_token;
      this.tokenCacheExpiry = newTokenData.expires_at - 300000;

      console.log('✅ Token refreshed successfully');
      return newTokenData;
    } catch (error: any) {
      console.error('❌ Failed to refresh token:', error.response?.data || error.message);
      console.log('🔐 Will need to reauthorize...');
      return null;
    }
  }

  /**
   * Get valid access token
   */
  async getValidToken(): Promise<string> {
    // Check cached token first
    if (this.cachedToken && this.tokenCacheExpiry && Date.now() < this.tokenCacheExpiry) {
      return this.cachedToken;
    }

    // Check if we have a saved token
    if (fs.existsSync(this.tokenPath)) {
      const tokenData: TokenData = JSON.parse(fs.readFileSync(this.tokenPath, 'utf-8'));

      // Check if token is still valid (with 5 minute buffer)
      if (tokenData.expires_at && tokenData.expires_at > (Date.now() + 300000)) {
        console.log('✅ Using existing valid token');
        this.cachedToken = tokenData.access_token;
        this.tokenCacheExpiry = Date.now() + 300000;
        return tokenData.access_token;
      }

      console.log('⏰ Token expired, attempting to refresh...');
      if (tokenData.refresh_token) {
        const refreshedToken = await this.refreshToken(tokenData);
        if (refreshedToken) {
          return refreshedToken.access_token;
        }
      }
    }

    // Need fresh authorization - check if already in progress
    if (this.authInProgress) {
      console.log('⏳ Authorization already in progress, waiting...');
      const newTokenData = await this.authInProgress;
      return newTokenData.access_token;
    }

    // Start new authorization
    console.log('🔐 No valid token found, starting authorization flow...');
    this.authInProgress = this.authorize();
    
    try {
      const newTokenData = await this.authInProgress;
      return newTokenData.access_token;
    } finally {
      this.authInProgress = null;
    }
  }

  /**
   * Download file with OAuth authentication
   */
  async downloadFile(downloadUrl: string, filePath: string): Promise<boolean> {
    const maxRetries = 3;
    let retryCount = 0;

    while (retryCount <= maxRetries) {
      try {
        const accessToken = await this.getValidToken();

        const response = await axios.get(downloadUrl, {
          headers: {
            'Authorization': `Bearer ${accessToken}`
          },
          responseType: 'stream',
          timeout: 90000
        });

        const writer = fs.createWriteStream(filePath);
        response.data.pipe(writer);

        await new Promise<void>((resolve, reject) => {
          writer.on('finish', resolve);
          writer.on('error', reject);
          response.data.on('error', reject);
        });

        return true;
      } catch (error: any) {
        retryCount++;
        
        // Clean up partial download
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }

        const isRetryable = error.code === 'ECONNABORTED' ||
          error.response?.status === 504 ||
          error.response?.status === 502 ||
          error.response?.status === 503 ||
          error.response?.status === 429;

        if (isRetryable && retryCount <= maxRetries) {
          const backoffDelay = Math.min(2000 * Math.pow(2, retryCount - 1), 30000);
          console.log(`⏳ Retry ${retryCount}/${maxRetries} in ${backoffDelay / 1000}s...`);
          await new Promise(resolve => setTimeout(resolve, backoffDelay));
        } else {
          throw error;
        }
      }
    }

    throw new Error(`Download failed after ${maxRetries} retries`);
  }
}
