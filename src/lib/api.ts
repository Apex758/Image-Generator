import axios from 'axios';

const apiClient = axios.create({
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
  generation_method?: string;
}

export interface ImagePrompt {
  prompt: string;
  explanation: string;
}

export interface LessonPlanAnalysisResponse {
  image_prompts: ImagePrompt[];
}

export interface ConfigData {
  use_hf_api: boolean;
  hf_api_configured: boolean;
  generation_method: string;
}

export interface Item {
  id: number;
  name: string;
  description?: string;
}

// SVG-related interfaces
export interface SVGTemplate {
  id: string;
  name: string;
  content_type: 'image_comprehension' | 'comic' | 'math' | 'worksheet';
  preview_url?: string;
  description?: string;
}

export interface GenerateSVGRequest {
  content_type: 'image_comprehension' | 'comic' | 'math' | 'worksheet';
  subject: string; // Must be one of: Mathematics, Language Arts, Science, Social Studies
  topic: string;
  grade_level: string; // Must be one of: Kindergarten, Grade 1-6
  layout_style?: string; // layout1, layout2, layout3
  num_questions?: number; // Will be passed to AI
  question_types?: string[]; // Will be passed to AI
  image_format?: string; // landscape, square, portrait, wide
  image_aspect_ratio?: { width: number; height: number }; // For AI image generation
  image_count?: number;
  custom_instructions?: string;
  include_activity_box?: boolean; // Toggle for activity drawing box
  include_word_bank?: boolean; // NEW: Toggle for word bank (fill-in-the-blank)
  max_pages?: number; // Maximum pages (for future multi-page support)
}

export interface GenerateSVGResponse {
  svg_content: string;
  template_id: string;
  placeholders: string[];
  images_generated: string[];  // Changed from 'images' to match backend field name
}

export interface ProcessSVGRequest {
  svg_content: string;
  text_replacements: Record<string, string>;  // Changed from 'replacements' to match backend
  add_writing_lines?: boolean;  // Added optional field that backend expects
}

export interface ProcessSVGResponse {
  processed_svg: string;  // Changed from svg_content to match backend
  replaced_placeholders: string[];  // Changed from processed_placeholders to match backend
}

export interface ExportSVGRequest {
  svg_content: string;
  format: 'pdf' | 'docx' | 'png';
  filename?: string;
}

export interface ExportSVGResponse {
  download_url: string;
  filename: string;
  format: string;
}

export interface SVGData {
  id: string;
  filename: string;
  url: string;
  template_id: string;
  created_at: string;
}

export const imageApi = {
  generate: async (request: GenerateImageRequest): Promise<ImageData> => {
    const response = await apiClient.post('/generate', request);
    return response.data;
  },

  list: async (): Promise<ImageData[]> => {
    const response = await apiClient.get('/images');
    return response.data;
  },

  delete: async (imageId: string): Promise<void> => {
    await apiClient.delete(`/images/${imageId}`);
  },

  getConfig: async (): Promise<ConfigData> => {
    const response = await apiClient.get('/config');
    return response.data;
  },

  analyzeLessonPlan: async (lessonPlan: string, maxImages: number = 5): Promise<LessonPlanAnalysisResponse> => {
    const response = await apiClient.post('/analyze-lesson-plan', {
      lesson_plan: lessonPlan,
      max_images: maxImages
    });
    return response.data;
  },
};

export const svgApi = {
  listTemplates: async (): Promise<SVGTemplate[]> => {
    const response = await apiClient.get('/svg-templates');
    return response.data;
  },

  generate: async (request: GenerateSVGRequest): Promise<GenerateSVGResponse> => {
    const response = await apiClient.post('/generate-svg', request);
    return response.data;
  },

  process: async (request: ProcessSVGRequest): Promise<ProcessSVGResponse> => {
    const response = await apiClient.post('/process-svg', request);
    return response.data;
  },

  export: async (request: ExportSVGRequest): Promise<ExportSVGResponse> => {
    const response = await apiClient.post('/export-svg', request);
    return response.data;
  },

  list: async (): Promise<SVGData[]> => {
    const response = await apiClient.get('/svg-items');
    return response.data;
  },

  delete: async (svgId: string): Promise<void> => {
    await apiClient.delete(`/svg-items/${svgId}`);
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