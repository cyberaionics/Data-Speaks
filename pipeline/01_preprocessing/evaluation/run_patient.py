from pathlib import Path
import json, time, traceback
import numpy as np

from baseline.stage0 import load_and_select_18

def run_one_recording(edf_path, pipeline_fn, output_dir, pipeline_id):
    edf_path = Path(edf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rec_id = edf_path.stem
    record = {
        "recording_id": rec_id, "file_path": str(edf_path),
        "status": "pending", "error": None,
        "output_path": None, "provenance_path": None,
        "n_channels": None, "n_samples": None,
        "sfreq_hz": None, "runtime_seconds": None
    }
    t0 = time.time()
    try:
        raw_std, s0_info = load_and_select_18(edf_path)
        raw_out, prov = pipeline_fn(raw_std, recording_id = rec_id)
        data = raw_out.get_data().astype(np.float32)
        out_npy = output_dir / f"{rec_id}_data.npy"
        np.save(out_npy, data)
        prev_full = {"pipeline_id": pipeline_id, "recording_id":rec_id, "file_path": str(edf_path), "stage0": s0_info, "candidate":prov}
        out_prov = output_dir / f"{rec_id}_provenence.json"
        out_prov.write_text(json.dumps(prov_full, indent=2, default=str))
        record.update({"status":"success", "output_path":str(out_npy), "provenence_path":str(out_prov), n_channels:int(data.shape[0]), "n_samples":int(data.shape[1]), "sfreq_hz":float(raw_out.info["sfreq"]),"runtime_seconds":round(time.time()-t0, 2)})
        print(f"    OK  {rec_id}    ({data.shape[0]} ch, {data.shape[1]} samp, " f"{record['runtime_seconds']}s)")
    except Exception as e:
        record.update({"status":"failure", "error":f"{type(e).__name__}: {e}", "traceback":traceback.format_exc(), "runtime_seconds":round(time.time()-t0, 2)})
        print(f"    FAIL    {rec_id}    ({type(e).__name__}: {str(e)[:80]})")
    return record


def run_patient(patient_dir, pipeline_fn, pipeline_id, output_root, summary_path):
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name
    output_dir = Path(output_root) / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)
    edfs = sorted(patient_dir.glob("*.edf"))
    print(f"\nPatient {patient_id} : {len(edfs)} EDFs -> {output_dir}")
    print("-"*60)
    records = []
    t0 = time.time()
    for edf in edfs:
        records.append(run_one_recording(edf, pipeline_fn, output_dir, pipeline_id))
    total = round(time.time()-t0, 2)
    n_ok = sum(r["status"] == "success" for r in records)
    n_fail = len(records) - n_ok
    summary = {"pipeline_id":pipeline_id, "patient_id":patient_id, "n_total":len(records), "n_success":n_ok, "n_failed":n_fail, "total_runtime_seconds":total, "records":records}
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print("-" * 60)
    print(f"Done. {n_ok} ok, {n_fail} failed. Total {total}s")
    print(f"Summary: {summary_path}")
    return summary


