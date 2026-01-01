# Runalyzer

Analisi rapida delle corse Garmin partendo da file **FIT** e **GPX** (con CSV intervalli opzionale) per generare report HTML/PNG/CSV riassuntivi.

## Requisiti
- Python 3.10+
- Dipendenze: `pip install -r requirements.txt` (oppure installa i pacchetti visti negli import del file `runalyzer.py`).

## Input supportati
- `BASE.fit`: dati principali (HR, power, gps, laps, training effect, peso se presente).
- `BASE.gpx`: fallback/integrazione per coordinate/altitudine se mancanti nel FIT.
- `BASE.csv` (opzionale): esportazione Garmin “Ripetute” per usare intervalli ufficiali.

### Cosa succede se manca un file?
- Senza **GPX** perdi solo il fallback per lat/lon/alt in caso siano assenti nel FIT.
- Senza **CSV** gli intervalli vengono ricavati da laps FIT (se presenti) o da un’autodetection.

## Esecuzione veloce
```bash
python runalyzer.py BASE \
  --indir /path/ai/file \
  --outdir out_analysis \
  --cat SOG \
  --hr_thr 172 --pwr_thr 376 --pace_thr 5:36 \
  --weight 70 --rpe 6
```

`BASE` è il nome comune dei file (es. `20250101-SOG.fit` / `.gpx` / `.csv`). L’output finisce in `out_analysis/YYYYMMDD-CAT/` con:
- `report.html` + PNG dei grafici
- `analysis_summary.md/json`
- `records_1s.csv` e `intervals_used.csv`

## Nuove funzioni principali
- Gauge “universale” per le metriche nerd (stile Training Effect, con colori/range configurabili).
- Cardiac drift aggiustato per il passo: esclude ritmi più lenti di 7:30/km e usa i residui HR vs passo.
- Grafico FC vs passo a bucket di 30s (solo ritmi ≤ 7:30/km).

## Note
- L’opzione `--no_plots` salta la generazione dei PNG.
- Le soglie accettano passo nel formato `m:ss` (es. `5:36`).
