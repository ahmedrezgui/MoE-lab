export interface ModelInfo { name: string; config: { vocab_size:number; d_model:number; n_layers:number; n_heads:number; n_kv_heads:number; max_seq_len:number; moe: {n_experts:number;top_k:number;load_balance_coef:number}|null }; total_parameters:number; active_parameters_estimate:number; device:string }
export interface Routing { scope:string; text:string; tokens:{id:number;text:string}[]; layers:{layer:number;probabilities:number[][];selected:number[][];weights:number[][];counts:number[]}[] }
export interface Run { model:string; seed:number; validation_curve:{step:number;val_ce:number;val_perplexity:number}[]; final_val_ce:number; loop_seconds:number;loop_tokens_per_second:number;peak_allocated_mib:number|null;session_training_tokens:number }
export interface Experiments { source:string|null; protocol?:string; runs:Run[] }
export interface Generation {text:string; generated_tokens:number; finish_reason:string}
export async function api<T>(path:string, body?:unknown, signal?:AbortSignal):Promise<T> {
  const response = await fetch(`/api${path}`, {signal: signal ?? AbortSignal.timeout(180000), ...(body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})});
  if (!response.ok) { const data = await response.json().catch(()=>null); throw new Error(typeof data?.detail === 'string' ? data.detail : `Request failed (${response.status}). Check the API and your input.`); }
  return response.json();
}
