# GPU in affitto e modelli modificati: abliterare Mistral 7B nel cloud

> Il secondo capitolo del progetto Heretic: stessa tecnica, modello più grande, GPU a noleggio per 2 euro — e qualche lezione inaspettata su come funziona davvero l'allineamento dei modelli AI.

---

## Il punto di partenza

Nel primo esperimento avevo abliterato **TinyLlama 1.1B** sulla mia GPU locale — una RTX 3070 con 8GB di VRAM. Aveva funzionato bene: 200 trial di ottimizzazione in 15 minuti, risultato pubblicato su Hugging Face.

Il passo successivo naturale era scalare: provare con un modello più capace. Ho scelto **Mistral 7B Instruct v0.3**, uno dei modelli open source più usati al mondo — sette volte più grande di TinyLlama, ma soprattutto addestrato con un processo di allineamento molto più aggressivo.

Il problema immediato: Mistral 7B in piena precisione richiede ~14GB di VRAM. La mia GPU ne ha 8. In locale avrebbe richiesto quantizzazione a 4-bit (una compressione dei pesi che degrada la qualità) e circa 2.5 ore di elaborazione. Soluzione: GPU in affitto nel cloud.

---

## Vast.ai: una GPU RTX 4090 per ~1.50 euro

Ho usato **[Vast.ai](https://vast.ai)** — un marketplace dove privati e datacenter mettono a disposizione GPU inutilizzate a prezzi molto inferiori ai cloud tradizionali come AWS o Google Cloud.

Ho noleggiato un'istanza con una **RTX 4090 da 48GB di VRAM** a circa $1.13/ora. Per un'operazione che si è conclusa in circa 30 minuti di ottimizzazione effettiva, il costo totale è stato **inferiore a 2 euro**.

Il confronto con la mia macchina locale è significativo:

| | RTX 3070 (locale) | RTX 4090 (Vast.ai) |
|---|---|---|
| VRAM | 8 GB | 48 GB |
| Quantizzazione necessaria | Sì (4-bit) | No — modello intero in bfloat16 |
| Tempo per 200 trial | ~2.5 ore (stima) | **19 minuti e 21 secondi** |
| Costo | €0 | ~€1.50 |
| Qualità del modello risultante | Ridotta (dequantizzazione) | Massima |

Sull'istanza cloud, Mistral 7B girava in **bfloat16 nativo** — la stessa precisione con cui è stato addestrato — senza compromessi. Il download del modello (14.5GB) ha impiegato 1 minuto e 34 secondi a 153 MB/s. Tutto molto diverso da un laptop.

---

## Come funziona il processo (riepilogo tecnico)

Per chi non ha letto il primo articolo: **Heretic** è uno strumento open source che modifica il comportamento di un LLM senza riaddestrarlo. Il processo in breve:

1. **Carica il modello** dai server di Hugging Face
2. **Esegue forward pass** su due dataset: prompt innocui (`mlabonne/harmless_alpaca`) e prompt problematici (`mlabonne/harmful_behaviors`)
3. **Analizza le attivazioni interne** del transformer su entrambi i set per identificare la "direzione del rifiuto" nello spazio latente — cioè il pattern matematico associato alle risposte di rifiuto
4. **Ottimizza i parametri** con Optuna (ottimizzazione bayesiana, 200 trial)
5. **Applica la correzione via LoRA** — una modifica leggera e mirata ai pesi

Il risultato non è un modello retrainato. È lo stesso modello con la geometria interna leggermente riorientata.

---

## I risultati: Trial 173

Al termine dei 200 trial, Heretic presenta una lista di combinazioni Pareto-ottimali — quelle che minimizzano sia il numero di rifiuti che la divergenza dal modello originale.

Ho scelto il **Trial 173**:

| Metrica | Valore |
|---|---|
| Rifiuti su 100 prompt | **4/100** |
| KL divergence | **0.0606** |

La KL divergence misura quanto il modello si discosta dall'originale: valori sotto 0.5 indicano che le capacità generali sono intatte. 0.0606 è eccellente — meglio del risultato ottenuto con TinyLlama (0.0840).

In pratica: il modello rifiuta 4 prompt su 100 invece di ~37, e le sue capacità originali sono largamente preservate.

Il modello è pubblicato su Hugging Face: **[paoloronco/Mistral-7B-Instruct-v0.3-heretic](https://huggingface.co/paoloronco/Mistral-7B-Instruct-v0.3-heretic)**

---

## La lezione più interessante: l'abliterazione non è magia

Dopo aver testato il modello, ho scoperto qualcosa che non avevo anticipato: **Mistral 7B abliterato risponde in modo molto meno "libero" rispetto a TinyLlama abliterato**.

Il perché è tecnico ma importante da capire.

TinyLlama era addestrato principalmente su testo grezzo con un allineamento molto leggero. Quando si abliterava la direzione del rifiuto, sotto non c'era quasi nulla — il modello rispondeva liberamente.

Mistral 7B Instruct ha subito un **RLHF (Reinforcement Learning from Human Feedback) intensivo**: migliaia di esempi di feedback umano che hanno modellato non solo i rifiuti espliciti, ma anche il tono, lo stile, la tendenza a moralizzare e a inserire disclaimer. L'abliterazione rimuove il meccanismo del rifiuto diretto — ma i pattern comportamentali profondi rimangono nei pesi.

Risultato pratico: il modello non dice più "non posso aiutarti con questo", ma la risposta che fornisce è comunque moderata, annacquata, con tendenza all'auto-censura implicita.

Questo ha senso una volta capito come funziona il processo: **Heretic modifica la geometria latente, non riscrive il training**. Su modelli con RLHF leggero, basta. Su modelli con RLHF pesante, rimuove il sintomo ma non la causa.

È una distinzione che non trovi spiegata chiaramente quasi da nessuna parte, e che cambia il modo in cui si sceglie il modello di partenza per questo tipo di esperimento.

---

## I problemi tecnici incontrati (e risolti)

Il percorso non è stato lineare. Alcune difficoltà che vale la pena documentare:

### Il proxy del nodo Vast.ai

Il nodo che avevo noleggiato aveva una configurazione di rete non standard: tutto il traffico verso Hugging Face veniva intercettato da un proxy locale (`117.175.104.83:8081`), probabilmente un mirror per ridurre i costi di banda del provider.

Questo causava due problemi:
- Il **download di modelli non popolari** falliva con errore 401 (il mirror non li aveva in cache)
- L'**upload verso Hugging Face** falliva perché il proxy non supportava il nuovo protocollo di storage (Xet) usato da HF

Soluzione per il download: `export HF_ENDPOINT=https://huggingface.co` prima di lanciare qualsiasi comando.

Soluzione per l'upload: disabilitare il protocollo Xet e forzare LFS classico con `HF_HUB_DISABLE_XET=1`.

### Il formato del tokenizer

Heretic 1.3.0 (la versione installata sul cloud) salva il tokenizer con una classe non standard (`TokenizersBackend`) che le versioni locali di Transformers non riconoscono. Risolto modificando il `tokenizer_config.json` per usare `PreTrainedTokenizerFast`.

Piccolo ma utile: mostra quanto il versionamento degli strumenti AI cambi velocemente e quanto sia importante documentare l'ambiente esatto usato per ogni esperimento.

---

## Cosa ho imparato (questa volta)

### Sul cloud computing per ML

Il cloud GPU ha senso economico per task puntuali e computazionalmente pesanti. Per addestrare o modificare modelli da 7B in su, noleggiare 30 minuti di RTX 4090 costa meno dell'elettricità che consumeresti a far girare la stessa operazione per ore sulla tua GPU locale — e il risultato è di qualità superiore perché si usa piena precisione senza quantizzazione.

Vast.ai in particolare è interessante perché il mercato è competitivo: i prezzi variano molto tra i nodi, e con un po' di attenzione si trovano istanze verificate a meno di $0.40/ora.

### Sul processo di allineamento dei modelli

L'esperimento ha reso concreto qualcosa che prima capivo solo in modo astratto: l'allineamento di un LLM non è uno strato separabile — è distribuito nei pesi attraverso tutto il training. Rimuovere il rifiuto esplicito è relativamente facile. Rimuovere i pattern comportamentali profondi richiede interventi diversi, più invasivi.

I modelli con RLHF leggero (come certi fine-tune community, o i modelli base senza instruction tuning) rispondono molto meglio all'abliterazione. I modelli con RLHF pesante (Meta Llama, modelli Instruct professionali) cedono il rifiuto esplicito ma mantengono il "carattere" del training.

### Sulla scelta del modello di partenza

Non tutti i modelli sono equivalenti come candidati per l'abliterazione. Il criterio più importante non è la dimensione — è **quanto aggressivo è stato il processo di allineamento**. Un 7B con RLHF leggero darà risultati più netti di un 7B con RLHF intensivo.

---

## Stack tecnico (aggiornato)

| Strumento | Ruolo |
|---|---|
| Heretic v1.3.0 | Tool di abliterazione |
| PyTorch 2.11 + CUDA 13.0 | Calcolo su GPU (cloud) |
| Transformers 5.9.0 | Caricamento modelli |
| bitsandbytes 0.49.2 | Quantizzazione (uso locale) |
| Optuna 4.8.0 | Ottimizzazione bayesiana |
| PEFT / LoRA | Applicazione modifiche ai pesi |
| Safetensors | Formato salvataggio modello |
| HuggingFace Hub | Pubblicazione e distribuzione |
| **Vast.ai** | **GPU cloud a noleggio (RTX 4090, 48GB VRAM)** |
| Gradio | Interfaccia chat locale per test |

---

## Cosa NON è questo progetto (ancora)

Come nel primo articolo: non sono un AI engineer. Ho usato strumenti esistenti, documentazione, e la capacità di capire cosa stavo facendo prima di farlo — inclusi i problemi tecnici che si presentano quando si esce dai tutorial e si lavora con ambienti reali.

L'errore più utile di questa sessione è stato probabilmente quello del proxy: ha costretto a capire come funziona il routing delle richieste HTTP in un ambiente cloud, come Hugging Face gestisce i diversi protocolli di upload, e come si diagnostica un problema di rete quando l'errore che vedi non corrisponde al problema reale.

---

## Link

- Modello pubblicato: [paoloronco/Mistral-7B-Instruct-v0.3-heretic](https://huggingface.co/paoloronco/Mistral-7B-Instruct-v0.3-heretic)
- Modello precedente: [paoloronco/TinyLlama-1.1B-Chat-v1.0-heretic](https://huggingface.co/paoloronco/TinyLlama-1.1B-Chat-v1.0-heretic)
- Repository GitHub modelli: [github.com/paoloronco/heretic-models](https://github.com/paoloronco/heretic-models)
- Heretic (tool originale): [github.com/p-e-w/heretic](https://github.com/p-e-w/heretic)
- Vast.ai (GPU cloud): [vast.ai](https://vast.ai)
- Profilo HuggingFace: [huggingface.co/paoloronco](https://huggingface.co/paoloronco)
