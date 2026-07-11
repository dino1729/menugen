import React from 'react';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi, type Mock } from 'vitest';
import App from './App';

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  close = vi.fn();

  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }
}

const configResponse = {
  image_provider: 'litellm',
  vision_model: 'gemini-3.5-flash',
  image_gen_model: 'gemini-3-pro-image-preview',
  video_gen_model: 'veo-3.1-generate-001',
  description_model: 'gemini-3.5-flash',
  whitelists: {
    vision: ['gemini-3.5-flash'],
    image: ['gemini-3-pro-image-preview'],
    video: ['veo-3.1-generate-001'],
    text: ['gemini-3.5-flash'],
  },
};

beforeEach(() => {
  MockWebSocket.instances = [];
  Object.defineProperty(globalThis, 'WebSocket', {
    configurable: true,
    writable: true,
    value: MockWebSocket,
  });
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith('/config')) {
      return { ok: true, json: async () => configResponse } as Response;
    }
    if (url.endsWith('/upload_menu/')) {
      return {
        ok: true,
        json: async () => ({ status: 'processing', sessionId: 'session-1' }),
      } as Response;
    }
    return { ok: true, json: async () => ({ status: 'cancelled' }) } as Response;
  }) as Mock;
});

afterEach(() => {
  cleanup();
  window.localStorage.clear();
});

test('shows image generation failures received over the session socket', async () => {
  const { container } = render(<App />);
  await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());

  const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
  fireEvent.change(fileInput, {
    target: { files: [new File(['image'], 'menu.jpg', { type: 'image/jpeg' })] },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Parse & Generate Images' }));

  await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));
  act(() => {
    MockWebSocket.instances[0].onmessage?.({
      data: JSON.stringify({
        type: 'image_generation_failed',
        item: 'Taco',
        message: 'provider unavailable',
      }),
    } as MessageEvent);
  });

  expect(
    screen.getAllByText(/Image generation failed for Taco: provider unavailable/)
  ).not.toHaveLength(0);

  act(() => {
    MockWebSocket.instances[0].onmessage?.({
      data: JSON.stringify({ type: 'done' }),
    } as MessageEvent);
  });

  expect(screen.queryByText('All images generated!')).not.toBeInTheDocument();
  expect(screen.getByText('Generation completed with errors')).toBeInTheDocument();

  act(() => {
    MockWebSocket.instances[0].onclose?.({ code: 1000, reason: '' } as CloseEvent);
  });
  expect(screen.getByText('Generation completed with errors')).toBeInTheDocument();
});

test('marks paid API requests as trusted frontend requests', async () => {
  const { container } = render(<App />);
  await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());

  const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
  fireEvent.change(fileInput, {
    target: { files: [new File(['image'], 'menu.jpg', { type: 'image/jpeg' })] },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Parse & Generate Images' }));

  await waitFor(() => {
    const uploadCall = (globalThis.fetch as Mock).mock.calls.find(
      ([url]) => String(url).endsWith('/upload_menu/')
    );
    expect(uploadCall?.[1]?.headers).toEqual({ 'X-MenuGen-Request': '1' });
  });
});

test('stop generation cancels the backend session', async () => {
  const { container } = render(<App />);
  await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());

  const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
  fireEvent.change(fileInput, {
    target: { files: [new File(['image'], 'menu.jpg', { type: 'image/jpeg' })] },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Parse & Generate Images' }));
  await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));

  fireEvent.click(screen.getByRole('button', { name: 'Stop Generating' }));

  await waitFor(() => {
    const cancelCall = (globalThis.fetch as Mock).mock.calls.find(
      ([url]) => String(url).endsWith('/sessions/session-1')
    );
    expect(cancelCall?.[1]).toMatchObject({
      method: 'DELETE',
      headers: { 'X-MenuGen-Request': '1' },
    });
  });
  expect(MockWebSocket.instances[0].close).toHaveBeenCalled();
});

test('rejects files larger than ten megabytes before upload', async () => {
  const { container } = render(<App />);
  await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
  const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
  const oversized = new File(
    [new Uint8Array((10 * 1024 * 1024) + 1)],
    'large.jpg',
    { type: 'image/jpeg' }
  );

  fireEvent.change(fileInput, { target: { files: [oversized] } });

  expect(screen.getByText('Image must be 10 MB or smaller.')).toBeInTheDocument();
  expect(screen.queryByText('Uploaded Menu Image')).not.toBeInTheDocument();
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

test('advances progress when each image finishes', async () => {
  const { container } = render(<App />);
  await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
  const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
  fireEvent.change(fileInput, {
    target: { files: [new File(['image'], 'menu.jpg', { type: 'image/jpeg' })] },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Parse & Generate Images' }));
  await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));

  act(() => {
    MockWebSocket.instances[0].onmessage?.({
      data: JSON.stringify({
        type: 'menu_parsed',
        data: { items: [{ name: 'Taco' }, { name: 'Soup' }] },
      }),
    } as MessageEvent);
    MockWebSocket.instances[0].onmessage?.({
      data: JSON.stringify({
        type: 'image_generated',
        item: 'Taco',
        url: '/images/session-1/000_Taco.png',
      }),
    } as MessageEvent);
  });

  expect(screen.getByText('65%')).toBeInTheDocument();
});

test('keeps duplicate dish names mapped to their own generated images', async () => {
  const { container } = render(<App />);
  await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
  const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
  fireEvent.change(fileInput, {
    target: { files: [new File(['image'], 'menu.jpg', { type: 'image/jpeg' })] },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Parse & Generate Images' }));
  await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));

  act(() => {
    MockWebSocket.instances[0].onmessage?.({
      data: JSON.stringify({
        type: 'menu_parsed',
        data: { items: [{ name: 'Taco' }, { name: 'Taco' }] },
      }),
    } as MessageEvent);
    MockWebSocket.instances[0].onmessage?.({
      data: JSON.stringify({
        type: 'image_generated', index: 0, item: 'Taco', url: '/images/s/000_Taco.png',
      }),
    } as MessageEvent);
    MockWebSocket.instances[0].onmessage?.({
      data: JSON.stringify({
        type: 'image_generated', index: 1, item: 'Taco', url: '/images/s/001_Taco.png',
      }),
    } as MessageEvent);
  });

  const dishImages = screen.getAllByAltText('Taco') as HTMLImageElement[];
  expect(dishImages).toHaveLength(2);
  expect(dishImages.map(image => image.src)).toEqual([
    'http://localhost:8005/images/s/000_Taco.png',
    'http://localhost:8005/images/s/001_Taco.png',
  ]);
});

test('reports an unexpected socket close before completion', async () => {
  const { container } = render(<App />);
  await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
  const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
  fireEvent.change(fileInput, {
    target: { files: [new File(['image'], 'menu.jpg', { type: 'image/jpeg' })] },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Parse & Generate Images' }));
  await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));

  act(() => {
    MockWebSocket.instances[0].onclose?.({ code: 1006, reason: '' } as CloseEvent);
  });

  expect(screen.getByText('WebSocket closed unexpectedly')).toBeInTheDocument();
});
