import { ConfigProvider, Switch, theme } from 'antd';
import { useEffect, useState } from 'react';

export default function ThemeControl({
  darkLabel,
  lightLabel,
}: {
  darkLabel: string;
  lightLabel: string;
}) {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    const saved = window.localStorage.getItem('typecv-theme');
    const initial =
      saved === 'dark' || (saved !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches);
    setDark(initial);
    document.documentElement.dataset.theme = initial ? 'dark' : 'light';
  }, []);
  const change = (value: boolean) => {
    setDark(value);
    document.documentElement.dataset.theme = value ? 'dark' : 'light';
    window.localStorage.setItem('typecv-theme', value ? 'dark' : 'light');
  };
  return (
    <ConfigProvider
      theme={{
        algorithm: dark ? theme.darkAlgorithm : theme.defaultAlgorithm,
        token: { colorPrimary: dark ? '#4096ff' : '#1677ff', borderRadius: 6 },
      }}
    >
      <Switch
        checked={dark}
        onChange={change}
        aria-label={dark ? lightLabel : darkLabel}
        size="small"
      />
    </ConfigProvider>
  );
}
