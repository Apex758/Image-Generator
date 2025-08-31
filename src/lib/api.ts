import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
});

export interface GenerateImageRequest {
  prompt: string;
  width?: number;
  height?: number;
  guidance_scale?: number;
  num_inference_steps?: number;
  seed?: number;
}

export interface ImageData {
  id: string;
  filename: string;
  url: string;
  prompt: string;
  created_at: string;
  width: number;
  height: number;
  seed?: number;
  guidance_scale?: number;
  num_inference_steps?: number;
}

export interface Item {
  id: number;
  name: string;
  description?: string;
}

export const imageApi = {
  generate: async (request: GenerateImageRequest): Promise<ImageData> => {
    const response = await api.post('/generate', request);
    return response.data;
  },

  list: async (): Promise<ImageData[]> => {
    const response = await api.get('/images');
    return response.data;
  },

  delete: async (imageId: string): Promise<void> => {
    await api.delete(`/images/${imageId}`);
  },
};

// Legacy API for the demo items
export const itemsApi = {
  list: async (): Promise<Item[]> => {
    // Mock data for now
    return [];
  },

  create: async (item: { name: string; description?: string }): Promise<Item> => {
    // Mock implementation
    return {
      id: Date.now(),
      name: item.name,
      description: item.description,
    };
  },
};

// Export legacy structure for backward compatibility
export const api_legacy = {
  items: itemsApi,
};

export { imageApi as api };