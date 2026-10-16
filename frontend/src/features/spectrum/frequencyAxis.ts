// The receiver averages raw correlator channels [600, 3672) in groups of four.
// Use bin edges around the averaged centres; 15–85 MHz is a displayed crop, not
// the frequency span of all 768 array rows. See stream_receiver.process_frame.
export const RAW_CHANNEL_MHZ=196/8192;
export const CHANNEL_MHZ=4*RAW_CHANNEL_MHZ;
export const FREQUENCY_MIN=(600-.5)*RAW_CHANNEL_MHZ;
export const FREQUENCY_MAX=FREQUENCY_MIN+768*CHANNEL_MHZ;
export const FREQUENCY_SPAN=FREQUENCY_MAX-FREQUENCY_MIN;
export function frequencyChannel(frequency:number) {
 return Math.max(0,Math.min(767,Math.floor((frequency-FREQUENCY_MIN)/CHANNEL_MHZ)));
}
export function frequencyCrop(fmin:number,fmax:number) {
 return {top:(FREQUENCY_MAX-fmax)/FREQUENCY_SPAN*768,height:(fmax-fmin)/FREQUENCY_SPAN*768};
}
