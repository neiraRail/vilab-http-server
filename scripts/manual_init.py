#!/usr/bin/env python3
"""Utilidad para ejecutar manualmente el flujo de inicialización.

Se proporciona uno o más identificadores de *Job* y se ejecutará el flujo de
aprendizaje inicial utilizando el último ``JobRun`` disponible para cada uno.
"""

import argparse
from bson import ObjectId

from app import app, mongo, run_initial_learning


def process_job(job_id: str) -> None:
    job = mongo.db.jobs.find_one({"_id": ObjectId(job_id)})
    if not job:
        print(f"Job {job_id} no encontrado")
        return

    jobrun = (
        mongo.db.jobruns.find({"j": ObjectId(job_id)})
        .sort("dt", -1)
        .limit(1)
    )
    jobrun_list = list(jobrun)
    if not jobrun_list:
        print(f"No existen JobRuns para el Job {job_id}")
        return

    jr_id = jobrun_list[0]["_id"]
    print(f"Ejecutando flujo de inicialización para Job {job_id} (JobRun {jr_id})")
    run_initial_learning(job, jr_id)
    print("Flujo de inicialización completado")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecutar el flujo de inicialización a partir de uno o más Job ID"
    )
    parser.add_argument("job_ids", nargs="+", help="Identificadores de los jobs")
    args = parser.parse_args()

    with app.app_context():
        for jid in args.job_ids:
            process_job(jid)


if __name__ == "__main__":
    main()
