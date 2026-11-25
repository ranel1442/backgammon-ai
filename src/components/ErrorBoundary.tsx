import React from 'react';
import { AlertTriangle, X } from 'lucide-react';

interface ErrorBoundaryProps {
  children: React.ReactNode;
  onClose?: () => void;
  fallbackTitle?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  errorMessage: string;
}

export class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      errorMessage: '',
    };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      errorMessage: error.message || 'אירעה שגיאה בלתי צפויה.',
    };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    console.error('[ErrorBoundary] Caught component error:', error, info);
  }

  private handleClose = () => {
    this.setState({ hasError: false, errorMessage: '' });
    this.props.onClose?.();
  };

  render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <div
          dir="rtl"
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/85 backdrop-blur-md p-4"
        >
          <div className="w-full max-w-md rounded-2xl bg-slate-900 border border-rose-500/40 shadow-2xl p-5 space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-2.5 text-rose-300">
                <AlertTriangle className="w-5 h-5 shrink-0" />
                <h2 className="text-sm font-bold text-white">
                  {this.props.fallbackTitle ?? 'אירעה שגיאה בחלון הסריקה'}
                </h2>
              </div>
              <button
                type="button"
                onClick={this.handleClose}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors cursor-pointer"
                title="סגור"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-rose-200/90 bg-rose-500/10 border border-rose-500/25 rounded-xl p-3 break-words">
              {this.state.errorMessage}
            </p>

            <div className="flex justify-end">
              <button
                type="button"
                onClick={this.handleClose}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-100 text-xs font-bold border border-slate-700 transition-colors cursor-pointer"
              >
                סגור וחזור למשחק
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
