import { DefaultTheme } from 'react-native-paper';

export const theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: '#2E8B57', // Sea Green
    accent: '#FFA500', // Orange
    background: '#F5F5F5',
    surface: '#FFFFFF',
    text: '#333333',
    success: '#28A745',
    warning: '#FFC107',
    error: '#DC3545',
    info: '#17A2B8',
    onPrimary: '#FFFFFF',
    onSurface: '#333333',
    onBackground: '#333333',
  },
  roundness: 12,
} as const;