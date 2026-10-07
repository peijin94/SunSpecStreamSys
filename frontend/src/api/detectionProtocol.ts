export interface Detection {
  id: number;
  class_id: number;
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
  timestamp: number;
  time_start?: number;
  time_end?: number;
  boot_id?: string;
  frame_sequence?: number;
}
