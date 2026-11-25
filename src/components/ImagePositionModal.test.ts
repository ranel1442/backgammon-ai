import React from 'react';
import { renderToString } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { createInitialBoard } from '../engine/rules';
import { ErrorBoundary } from './ErrorBoundary';
import { ImagePositionModal } from './ImagePositionModal';
import modalSource from './ImagePositionModal.tsx?raw';

describe('ImagePositionModal Rules of Hooks & ErrorBoundary regression', () => {
  it('declares all React hooks strictly before the `if (!isOpen) return null;` guard', () => {
    const guardIndex = modalSource.indexOf('if (!isOpen) return null;');
    expect(guardIndex).toBeGreaterThan(-1);

    const afterGuard = modalSource.slice(guardIndex);
    const hookPattern =
      /\buse(?:State|Effect|Ref|Memo|Callback|Reducer|Context|LayoutEffect)\s*[<(]/;
    expect(hookPattern.test(afterGuard)).toBe(false);
  });

  it('renders cleanly across isOpen=false -> true -> false transitions without React Hook #310 errors', () => {
    const board = createInitialBoard();
    const onClose = vi.fn();
    const onConfirmPosition = vi.fn();

    const closedHtml1 = renderToString(
      React.createElement(ImagePositionModal, {
        isOpen: false,
        initialBoardState: board,
        onClose,
        onConfirmPosition,
      })
    );
    expect(closedHtml1).toBe('');

    const openHtml = renderToString(
      React.createElement(ImagePositionModal, {
        isOpen: true,
        initialBoardState: board,
        onClose,
        onConfirmPosition,
      })
    );
    expect(openHtml).toContain('סריקת לוח מתמונה');

    const closedHtml2 = renderToString(
      React.createElement(ImagePositionModal, {
        isOpen: false,
        initialBoardState: board,
        onClose,
        onConfirmPosition,
      })
    );
    expect(closedHtml2).toBe('');
  });

  it('ErrorBoundary renders Hebrew fallback UI when a child throws', () => {
    const derived = ErrorBoundary.getDerivedStateFromError(
      new Error('Simulated modal crash')
    );
    expect(derived.hasError).toBe(true);
    expect(derived.errorMessage).toBe('Simulated modal crash');
  });
});
