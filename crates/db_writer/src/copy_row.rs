use bytes::BytesMut;
use std::io;

pub const NULL: &[u8] = br"\N";
pub const TAB: u8 = b'\t';
pub const NL: u8 = b'\n';

/// Writer that appends into BytesMut without allocating.
pub struct BytesMutWriter<'a> {
    buf: &'a mut BytesMut,
}

impl<'a> BytesMutWriter<'a> {
    #[inline]
    pub fn new(buf: &'a mut BytesMut) -> Self {
        Self { buf }
    }
}

impl<'a> io::Write for BytesMutWriter<'a> {
    #[inline]
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// TSV row builder.
/// Usage:
///   let mut row = CopyRow::new(buf);
///   row.begin();
///   row.i64(123); row.f64(1.23); row.null(); row.end();
pub struct CopyRow<'a> {
    buf: &'a mut BytesMut,
    first: bool,
}

impl<'a> CopyRow<'a> {
    #[inline]
    pub fn new(buf: &'a mut BytesMut) -> Self {
        Self { buf, first: true }
    }

    #[inline]
    pub fn begin(&mut self) {
        self.first = true;
    }

    #[inline]
    fn sep(&mut self) {
        if self.first {
            self.first = false;
        } else {
            self.buf.extend_from_slice(&[TAB]);
        }
    }

    #[inline]
    pub fn end(&mut self) {
        self.buf.extend_from_slice(&[NL]);
    }

    #[inline]
    pub fn null(&mut self) {
        self.sep();
        self.buf.extend_from_slice(NULL);
    }

    // #[inline]
    // pub fn bytes(&mut self, b: &[u8]) {
    //     self.sep();
    //     self.buf.extend_from_slice(b);
    // }

    #[inline]
    pub fn str(&mut self, s: &str) {
        self.sep();
        self.buf.extend_from_slice(s.as_bytes());
    }

    #[inline]
    pub fn i64(&mut self, v: i64) {
        self.sep();
        let mut b = itoa::Buffer::new();
        self.buf.extend_from_slice(b.format(v).as_bytes());
    }

    // #[inline]
    // pub fn i16(&mut self, v: i16) {
    //     self.sep();
    //     let mut b = itoa::Buffer::new();
    //     self.buf.extend_from_slice(b.format(v).as_bytes());
    // }

    // #[inline]
    // pub fn u64(&mut self, v: u64) {
    //     self.sep();
    //     let mut b = itoa::Buffer::new();
    //     self.buf.extend_from_slice(b.format(v).as_bytes());
    // }

    #[inline]
    pub fn f64(&mut self, v: f64) {
        self.sep();
        let mut b = ryu::Buffer::new();
        // format_finite быстрее и без NaN/Inf (их не должно быть)
        self.buf.extend_from_slice(b.format_finite(v).as_bytes());
    }

    // #[inline]
    // pub fn opt_f64(&mut self, v: Option<f64>) {
    //     match v {
    //         Some(x) => self.f64(x),
    //         None => self.null(),
    //     }
    // }

    #[inline]
    pub fn opt_f32(&mut self, v: Option<f32>) {
        match v {
            Some(x) => self.f64(x as f64),
            None => self.null(),
        }
    }

    /// Writes JSON into buffer without allocating a String/Vec.
    #[inline]
    pub fn opt_json(&mut self, v: &Option<serde_json::Value>) -> Result<(), io::Error> {
        self.sep();
        match v {
            None => {
                self.buf.extend_from_slice(NULL);
                Ok(())
            }
            Some(j) => {
                let mut w = BytesMutWriter::new(self.buf);
                serde_json::to_writer(&mut w, j)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                Ok(())
            }
        }
    }
}
