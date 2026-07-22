use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use paw_core::Error;

use crate::models::QuantizedModel;

type LoadFn = Mutex<Box<dyn Fn() -> Result<Box<dyn QuantizedModel>, Error> + Send>>;

#[doc(hidden)]
pub struct PoolState {
    pub models: Vec<Arc<Mutex<Box<dyn QuantizedModel>>>>,
    pub used: usize,
    pub max: usize,
    loading: bool,
}

pub struct ModelPool {
    #[doc(hidden)]
    pub state: Mutex<PoolState>,
    cv: Condvar,
    load_fn: LoadFn,
}

pub(crate) struct PoolPermit<'a> {
    pool: &'a ModelPool,
}

impl ModelPool {
    pub(crate) fn acquire(&self) -> Result<PoolPermit<'_>, Error> {
        loop {
            let mut s = self.state.lock().unwrap();

            if s.used < s.models.len() {
                s.used += 1;
                return Ok(PoolPermit { pool: self });
            }

            if s.models.len() >= s.max || s.loading {
                s = self.cv.wait(s).unwrap();
                continue;
            }

            s.loading = true;
            drop(s);

            match self.load_and_push() {
                Ok(()) => return Ok(PoolPermit { pool: self }),
                Err(e) => {
                    let mut s = self.state.lock().unwrap();
                    s.loading = false;
                    self.cv.notify_one();
                    return Err(e);
                }
            }
        }
    }

    fn load_and_push(&self) -> Result<(), Error> {
        let new_model = (self.load_fn.lock().unwrap())()?;
        let model = Arc::new(Mutex::new(new_model));

        let mut s = self.state.lock().unwrap();
        s.loading = false;
        s.models.push(model);
        s.used += 1;

        let has_free = s.used < s.models.len();
        drop(s);

        if has_free {
            self.cv.notify_one();
        }
        Ok(())
    }
}

impl Drop for PoolPermit<'_> {
    fn drop(&mut self) {
        let mut s = self.pool.state.lock().unwrap();
        s.used = s.used.saturating_sub(1);
        self.pool.cv.notify_one();
    }
}

type PoolMap = HashMap<String, Arc<ModelPool>>;

static POOLS: OnceLock<Mutex<PoolMap>> = OnceLock::new();

fn pools() -> &'static Mutex<PoolMap> {
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn get_or_load_model(
    interpreter_key: &str,
    num_copies: usize,
    load_fn: impl Fn() -> Result<Box<dyn QuantizedModel>, Error> + Send + 'static,
) -> Result<(Arc<Mutex<Box<dyn QuantizedModel>>>, Arc<ModelPool>), Error> {
    let mut map = pools().lock().unwrap();
    if let Some(pool) = map.get(interpreter_key) {
        let s = pool.state.lock().unwrap();
        let model = Arc::clone(&s.models[0]);
        return Ok((model, Arc::clone(pool)));
    }
    let copies = if num_copies == 0 { 1 } else { num_copies };
    let first_model = Arc::new(Mutex::new(load_fn()?));

    let pool = Arc::new(ModelPool {
        state: Mutex::new(PoolState {
            models: vec![Arc::clone(&first_model)],
            used: 0,
            max: copies,
            loading: false,
        }),
        cv: Condvar::new(),
        load_fn: Mutex::new(Box::new(load_fn)),
    });

    map.insert(interpreter_key.to_string(), Arc::clone(&pool));

    Ok((first_model, pool))
}
