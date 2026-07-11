import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, test, vi, type Mock } from 'vitest';
import App from './App';

const configResponse = {
  image_provider: 'litellm',
  vision_model: 'gemini-3.5-flash',
  image_gen_model: 'gemini-3-pro-image-preview',
  video_gen_model: 'veo-3.1-generate-001',
  description_model: 'gemini-3.5-flash',
  whitelists: { vision: [], image: [], video: [], text: [] },
};

beforeEach(() => {
  globalThis.fetch = vi.fn(async () => ({
    ok: true,
    json: async () => configResponse,
  })) as Mock;
});

test('advertises clipboard paste in the upload box', async () => {
  render(<App />);
  await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());

  expect(screen.getByText(/paste an image/i)).toBeInTheDocument();
  expect(screen.getByText(/Ctrl\/⌘ \+ V/i)).toBeInTheDocument();
});

test('accepts an image pasted from the clipboard', async () => {
  render(<App />);
  await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
  const pastedImage = new File(['image'], 'pasted-menu.png', { type: 'image/png' });

  fireEvent.paste(document, {
    clipboardData: {
      files: [pastedImage],
      items: [],
    },
  });

  expect(await screen.findByText('Uploaded Menu Image')).toBeInTheDocument();
  expect(await screen.findByAltText('Uploaded Menu')).toBeInTheDocument();
});

test('rejects pasted image formats the backend does not support', async () => {
  render(<App />);
  await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
  const pastedImage = new File(['image'], 'pasted-menu.webp', { type: 'image/webp' });

  fireEvent.paste(document, {
    clipboardData: {
      files: [pastedImage],
      items: [],
    },
  });

  expect(screen.getByText('Upload a JPEG, PNG, or GIF image.')).toBeInTheDocument();
  expect(screen.queryByText('Uploaded Menu Image')).not.toBeInTheDocument();
});
