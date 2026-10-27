import { Button } from 'antd';

/**
 * The hardware check's entry in the lab's benchmark area: the run settings, under the standard
 * benchmark. It opens `/lab/check/` (`docs/lab-run-records.md`, "Hardware check").
 */
export default function HardwareCheckEntry({ ko }: { ko: boolean }) {
  return (
    <>
      <Button block href={ko ? '/ko/lab/check/' : '/lab/check/'}>
        {ko ? '하드웨어 점검' : 'Hardware check'}
      </Button>
      <p className="field-help">
        {ko
          ? '하드웨어 점검은 이 기기의 GPU에서 커널 다섯 묶음을 실행하고 CPU 기준과 비교합니다. 동의하면 점검 기록 하나를 보냅니다.'
          : 'The hardware check runs five kernel families on the GPU of this device and compares them with the CPU reference. With your consent it sends one check record.'}
      </p>
    </>
  );
}
