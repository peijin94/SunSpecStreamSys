import { radioImageUrl } from '../../api/client';
import { useRadioImages } from './useRadioImages';

const channels = [30, 45, 60, 75];

export function RadioImagesPanel() {
  const { metadata, images, error } = useRadioImages();
  const observed = metadata?.observation_time
    ? new Date(metadata.observation_time).toISOString().replace('T', ' ').replace('.000Z', ' UTC') : null;
  const stale = !!error || metadata?.status === 'stale';
  return (
    <section aria-labelledby="radio-images-title" className="radio-images-panel min-w-0 overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/80 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-4 py-3">
        <h2 id="radio-images-title" className="text-sm font-semibold">Live solar images · Stokes I</h2>
        <span className="text-xs text-slate-400">Updates every minute</span>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 pt-3 text-xs">
        <p className="text-slate-400">{observed ? `Observed ${observed}` : 'Waiting for the first observation'}</p>
        {images.length > 0 && <span className={stale ? 'text-amber-300' : 'text-emerald-400'}>
          {stale ? 'Last available images' : 'Latest observation'}
        </span>}
      </div>
      {error && <p role="status" className="px-4 pt-2 text-xs text-amber-300">{error}</p>}
      <div className="radio-images-grid grid min-w-0 grid-cols-1 gap-3 p-3 sm:grid-cols-2">
        {channels.map(target => {
          const image = images.find(item => item.target_mhz === target);
          return (
            <figure key={target} className="radio-image-tile min-w-0 overflow-hidden rounded-xl border border-slate-800 bg-[#0b1120]">
              <figcaption className="flex flex-wrap items-baseline justify-between gap-x-2 gap-y-1 px-3 pt-2 text-sm">
                <span className="font-medium">{target} MHz</span>
                {image && <span className="text-xs text-slate-400">Channel center {image.freq_mhz.toFixed(2)} MHz</span>}
              </figcaption>
              {image ? <a className="radio-image-content" href={radioImageUrl(image.image_url)} target="_blank" rel="noreferrer" aria-label={`Open ${target} MHz solar image`}>
                <img src={image.src} alt={`Stokes I solar image near ${target} MHz, observed ${observed}`} width={600} height={500}
                  className="block h-auto w-full" />
              </a> : <div className="radio-image-content radio-image-placeholder flex aspect-[6/5] items-center justify-center text-sm text-slate-500">Waiting for image</div>}
            </figure>
          );
        })}
      </div>
      <p className="px-4 pb-4 text-xs text-slate-400">Brightness temperature in MK · Shared color scale · Dashed circle: optical solar limb</p>
    </section>
  );
}
