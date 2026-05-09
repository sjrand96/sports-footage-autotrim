import { useMemo } from 'react'
import { computeTimelineMetrics } from './timelineMetrics.js'

function pctOfClip(sec, duration) {
  if (!Number.isFinite(duration) || duration <= 0) return '—'
  return `${((sec / duration) * 100).toFixed(1)}%`
}

function secLabel(sec) {
  return `${sec.toFixed(1)} s`
}

export default function MetricsPanel({ duration, intervals, groundTruthIntervals }) {
  const metrics = useMemo(() => {
    const pred = (intervals ?? []).map(({ start, end }) => ({ start, end }))
    const gt = (groundTruthIntervals ?? []).map(({ start, end }) => ({
      start,
      end,
    }))
    return computeTimelineMetrics(pred, gt, duration)
  }, [intervals, groundTruthIntervals, duration])

  const d = duration

  return (
    <aside className="metrics-panel" aria-labelledby="metrics-panel-title">
      <h2 id="metrics-panel-title" className="metrics-panel-title">
        Metrics
      </h2>

      <section className="metrics-panel-section">
        <h3 className="metrics-panel-heading">Timeline confusion</h3>
        <p className="metrics-panel-caption">
          Share of clip duration (% and seconds).
        </p>
        <div className="metrics-matrix-scroll">
          <table className="metrics-matrix">
            <thead>
              <tr>
                <th scope="col" className="metrics-matrix-corner" />
                <th scope="col" className="metrics-matrix-axis">
                  Predicted downtime
                </th>
                <th scope="col" className="metrics-matrix-axis">
                  Predicted playing
                </th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row" className="metrics-matrix-row-heading">
                  Actual downtime
                </th>
                <td className="metrics-matrix-cell metrics-matrix-cell--tn">
                  <span className="metrics-matrix-val">{pctOfClip(metrics.tnSec, d)}</span>
                  <span className="metrics-matrix-sub">{secLabel(metrics.tnSec)}</span>
                </td>
                <td className="metrics-matrix-cell metrics-matrix-cell--fp">
                  <span className="metrics-matrix-val">{pctOfClip(metrics.fpSec, d)}</span>
                  <span className="metrics-matrix-sub">{secLabel(metrics.fpSec)}</span>
                </td>
              </tr>
              <tr>
                <th scope="row" className="metrics-matrix-row-heading">
                  Actual playing
                </th>
                <td className="metrics-matrix-cell metrics-matrix-cell--fn">
                  <span className="metrics-matrix-val">{pctOfClip(metrics.fnSec, d)}</span>
                  <span className="metrics-matrix-sub">{secLabel(metrics.fnSec)}</span>
                </td>
                <td className="metrics-matrix-cell metrics-matrix-cell--tp">
                  <span className="metrics-matrix-val">{pctOfClip(metrics.tpSec, d)}</span>
                  <span className="metrics-matrix-sub">{secLabel(metrics.tpSec)}</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="metrics-panel-section">
        <h3 className="metrics-panel-heading">Playing coverage</h3>
        <dl className="metrics-coverage">
          <div className="metrics-coverage-row">
            <dt>Predicted</dt>
            <dd>{`${metrics.predictedCoveragePct.toFixed(1)}%`}</dd>
          </div>
          <div className="metrics-coverage-row">
            <dt>Actual</dt>
            <dd>{`${metrics.gtCoveragePct.toFixed(1)}%`}</dd>
          </div>
        </dl>
      </section>
    </aside>
  )
}
