// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// This file contains the implementation of Go's map type.
//
// A map is just a hash table. The data is arranged
// into an array of buckets. Each bucket contains up to
// 8 key/elem pairs. The low-order bits of the hash are
// used to select a bucket. Each bucket contains a few
// high-order bits of each hash to distinguish the entries
// within a single bucket.
//
// If more than 8 keys hash to a bucket, we chain on
// extra buckets.
//
// When the hashtable grows, we allocate a new array
// of buckets twice as big. Buckets are incrementally
// copied from the old bucket array to the new bucket array.
//
// Map iterators walk through the array of buckets and
// return the keys in walk order (bucket #, then overflow
// chain order, then bucket index).  To maintain iteration
// semantics, we never move keys within their bucket (if
// we did, keys might be returned 0 or 2 times).  When
// growing the table, iterators remain iterating through the
// old table and must check the new table if the bucket
// they are iterating through has been moved ("evacuated")
// to the new table.

// Picking loadFactor: too large and we have lots of overflow
// buckets, too small and we waste a lot of space. I wrote
// a simple program to check some stats for different loads:
// (64-bit, 8 byte keys and elems)
//  loadFactor    %overflow  bytes/entry     hitprobe    missprobe
//        4.00         2.13        20.77         3.00         4.00
//        4.50         4.05        17.30         3.25         4.50
//        5.00         6.85        14.77         3.50         5.00
//        5.50        10.55        12.94         3.75         5.50
//        6.00        15.27        11.67         4.00         6.00
//        6.50        20.90        10.79         4.25         6.50
//        7.00        27.14        10.15         4.50         7.00
//        7.50        34.03         9.73         4.75         7.50
//        8.00        41.10         9.40         5.00         8.00
//
// %overflow   = percentage of buckets which have an overflow bucket
// bytes/entry = overhead bytes used per key/elem pair
// hitprobe    = # of entries to check when looking up a present key
// missprobe   = # of entries to check when looking up an absent key
//
// Keep in mind this data is for maximally loaded tables, i.e. just
// before the table grows. Typical tables will be somewhat less loaded.

import (
	"runtime/internal/atomic"
	"runtime/internal/math"
	"runtime/internal/sys"
	"unsafe"
)

const (
	// Maximum number of key/elem pairs a bucket can hold.
	bucketCntBits = 3
	bucketCnt     = 1 << bucketCntBits

	// Maximum average load of a bucket that triggers growth is 6.5.
	// Represent as loadFactorNum/loadFactDen, to allow integer math.
	loadFactorNum = 13
	loadFactorDen = 2

	// Maximum key or elem size to keep inline (instead of mallocing per element).
	// Must fit in a uint8.
	// Fast versions cannot handle big elems - the cutoff size for
	// fast versions in cmd/compile/internal/gc/walk.go must be at most this elem.
	maxKeySize  = 128
	maxElemSize = 128

	// data offset should be the size of the bmap struct, but needs to be
	// aligned correctly. For amd64p32 this means 64-bit alignment
	// even though pointers are 32 bit.
	dataOffset = unsafe.Offsetof(struct {
		b bmap
		v int64
	}{}.v)

	// Possible tophash values. We reserve a few possibilities for special marks.
	// Each bucket (including its overflow buckets, if any) will have either all or none of its
	// entries in the evacuated* states (except during the evacuate() method, which only happens
	// during map writes and thus no one else can observe the map during that time).
	emptyRest      = 0 // this cell is empty, and there are no more non-empty cells at higher indexes or overflows.
	emptyOne       = 1 // this cell is empty

				       // 扩容时使用：key和value已经迁移到新的buckets中，但是处于新bucket的前半部分
	evacuatedX     = 2 // key/elem is valid.  Entry has been evacuated to first half of larger table.

					   // 同evacuatedX，但是key在后半部分
	evacuatedY     = 3 // same as above, but evacuated to second half of larger table.

					   // cell已经迁移到新的bucket
	evacuatedEmpty = 4 // cell is empty, bucket is evacuated.

	minTopHash     = 5 // minimum tophash for a normal filled cell.

	// flags
	iterator     = 1 // 可能有迭代器在使用 buckets ，there may be an iterator using buckets
	oldIterator  = 2 // 可能有迭代器在使用 oldbuckets，there may be an iterator using oldbuckets
	hashWriting  = 4 // 有协程正在向map中写入key，a goroutine is writing to the map
	sameSizeGrow = 8 // 等量扩容，the current map growth is to a new map of the same size

	// sentinel bucket ID for iterator checks
	noCheck = 1<<(8*sys.PtrSize) - 1
)

// isEmpty reports whether the given tophash array entry represents an empty bucket entry.
func isEmpty(x uint8) bool {
	return x <= emptyOne
}

// A header for a Go map.
type hmap struct {
	// Note: the format of the hmap is also encoded in cmd/compile/internal/gc/reflect.go.
	// Make sure this stays in sync with the compiler's definition.
	count     int // map元素数量 # live cells == size of map.  Must be first (used by len() builtin)
	flags     uint8
	B         uint8  // log_2 of # of buckets (can hold up to loadFactor * 2^B items)
					 // B表示对数，B决定了桶的数量，len(buckets) = 2^B, 即B=log2(len(buckets))
					 // map最多能够装载 loadFactor * 2^B，超出则会触发扩容

	noverflow uint16 // approximate number of overflow buckets; see incrnoverflow for details
	hash0     uint32 // hash seed hash随机种子

	buckets    unsafe.Pointer // array of 2^B Buckets. may be nil if count==0.
	oldbuckets unsafe.Pointer // previous bucket array of half the size, non-nil only when growing
	nevacuate  uintptr        // progress counter for evacuation (buckets less than this have been evacuated)

	extra *mapextra // optional fields
}

// mapextra holds fields that are not present on all maps.
type mapextra struct {
	// If both key and elem do not contain pointers and are inline, then we mark bucket
	// type as containing no pointers. This avoids scanning such maps.
	// However, bmap.overflow is a pointer. In order to keep overflow buckets
	// alive, we store pointers to all overflow buckets in hmap.extra.overflow and hmap.extra.oldoverflow.
	// overflow and oldoverflow are only used if key and elem do not contain pointers.
	// overflow contains overflow buckets for hmap.buckets.
	// oldoverflow contains overflow buckets for hmap.oldbuckets.
	// The indirection allows to store a pointer to the slice in hiter.
	overflow    *[]*bmap
	oldoverflow *[]*bmap

	// nextOverflow holds a pointer to a free overflow bucket.
	nextOverflow *bmap
}

// A bucket for a Go map.
type bmap struct {
	// tophash generally contains the top byte of the hash value
	// for each key in this bucket. If tophash[0] < minTopHash,
	// tophash[0] is a bucket evacuation state instead.
	tophash [bucketCnt]uint8
	// Followed by bucketCnt keys and then bucketCnt elems.
	// NOTE: packing all the keys together and then all the elems together makes the
	// code a bit more complicated than alternating key/elem/key/elem/... but it allows
	// us to eliminate padding which would be needed for, e.g., map[int64]int8.
	// Followed by an overflow pointer.
}

// A hash iteration structure.
// If you modify hiter, also change cmd/compile/internal/gc/reflect.go to indicate
// the layout of this structure.
type hiter struct {
	key         unsafe.Pointer // Must be in first position.  Write nil to indicate iteration end (see cmd/internal/gc/range.go).
	elem        unsafe.Pointer // Must be in second position (see cmd/internal/gc/range.go).
	t           *maptype
	h           *hmap
	buckets     unsafe.Pointer // bucket ptr at hash_iter initialization time
	bptr        *bmap          // current bucket
	overflow    *[]*bmap       // keeps overflow buckets of hmap.buckets alive
	oldoverflow *[]*bmap       // keeps overflow buckets of hmap.oldbuckets alive
	startBucket uintptr        // bucket iteration started at
	offset      uint8          // intra-bucket offset to start from during iteration (should be big enough to hold bucketCnt-1)
	wrapped     bool           // already wrapped around from end of bucket array to beginning
	B           uint8
	i           uint8
	bucket      uintptr
	checkBucket uintptr
}

// bucketShift returns 1<<b, optimized for code generation.
func bucketShift(b uint8) uintptr {
	// Masking the shift amount allows overflow checks to be elided.
	return uintptr(1) << (b & (sys.PtrSize*8 - 1))
}

// bucketMask returns 1<<b - 1, optimized for code generation.
func bucketMask(b uint8) uintptr {
	return bucketShift(b) - 1
}

// tophash calculates the tophash value for hash.
func tophash(hash uintptr) uint8 {
	top := uint8(hash >> (sys.PtrSize*8 - 8))
	if top < minTopHash {
		top += minTopHash
	}
	return top
}

func evacuated(b *bmap) bool {
	h := b.tophash[0]
	return h > emptyOne && h < minTopHash
}

func (b *bmap) overflow(t *maptype) *bmap {
	return *(**bmap)(add(unsafe.Pointer(b), uintptr(t.bucketsize)-sys.PtrSize))
}

func (b *bmap) setoverflow(t *maptype, ovf *bmap) {
	*(**bmap)(add(unsafe.Pointer(b), uintptr(t.bucketsize)-sys.PtrSize)) = ovf
}

func (b *bmap) keys() unsafe.Pointer {
	return add(unsafe.Pointer(b), dataOffset)
}

// incrnoverflow increments h.noverflow.
// noverflow counts the number of overflow buckets.
// This is used to trigger same-size map growth.
// See also tooManyOverflowBuckets.
// To keep hmap small, noverflow is a uint16.
// When there are few buckets, noverflow is an exact count.
// When there are many buckets, noverflow is an approximate count.
func (h *hmap) incrnoverflow() {
	// We trigger same-size map growth if there are
	// as many overflow buckets as buckets.
	// We need to be able to count to 1<<h.B.
	if h.B < 16 {
		h.noverflow++
		return
	}
	// Increment with probability 1/(1<<(h.B-15)).
	// When we reach 1<<15 - 1, we will have approximately
	// as many overflow buckets as buckets.
	mask := uint32(1)<<(h.B-15) - 1
	// Example: if h.B == 18, then mask == 7,
	// and fastrand & 7 == 0 with probability 1/8.
	if fastrand()&mask == 0 {
		h.noverflow++
	}
}

func (h *hmap) newoverflow(t *maptype, b *bmap) *bmap {
	var ovf *bmap
	if h.extra != nil && h.extra.nextOverflow != nil {
		// We have preallocated overflow buckets available.
		// See makeBucketArray for more details.
		ovf = h.extra.nextOverflow
		if ovf.overflow(t) == nil {
			// We're not at the end of the preallocated overflow buckets. Bump the pointer.
			h.extra.nextOverflow = (*bmap)(add(unsafe.Pointer(ovf), uintptr(t.bucketsize)))
		} else {
			// This is the last preallocated overflow bucket.
			// Reset the overflow pointer on this bucket,
			// which was set to a non-nil sentinel value.
			ovf.setoverflow(t, nil)
			h.extra.nextOverflow = nil
		}
	} else {
		ovf = (*bmap)(newobject(t.bucket))
	}
	h.incrnoverflow()
	if t.bucket.ptrdata == 0 {
		h.createOverflow()
		*h.extra.overflow = append(*h.extra.overflow, ovf)
	}
	b.setoverflow(t, ovf)
	return ovf
}

func (h *hmap) createOverflow() {
	if h.extra == nil {
		h.extra = new(mapextra)
	}
	if h.extra.overflow == nil {
		h.extra.overflow = new([]*bmap)
	}
}

func makemap64(t *maptype, hint int64, h *hmap) *hmap {
	if int64(int(hint)) != hint {
		hint = 0
	}
	return makemap(t, int(hint), h)
}

// makemap_small implements Go map creation for make(map[k]v) and
// make(map[k]v, hint) when hint is known to be at most bucketCnt
// at compile time and the map needs to be allocated on the heap.
func makemap_small() *hmap {
	h := new(hmap)
	h.hash0 = fastrand()
	return h
}

// makemap implements Go map creation for make(map[k]v, hint).
// If the compiler has determined that the map or the first bucket
// can be created on the stack, h and/or bucket may be non-nil.
// If h != nil, the map can be created directly in h.
// If h.buckets != nil, bucket pointed to can be used as the first bucket.
func makemap(t *maptype, hint int, h *hmap) *hmap {
	// hint，初始化map的容量
	// 判断是否溢出或者超出能分配的最大值
	mem, overflow := math.MulUintptr(uintptr(hint), t.bucket.size)
	if overflow || mem > maxAlloc {
		hint = 0 // hint设置为0
	}

	println("h=", h, "hint=", hint)
	// initialize Hmap
	if h == nil {
		h = new(hmap) // 构造一个hmap, new返回的是指针
	}
	h.hash0 = fastrand()// 获取一个随机种子
	// Find the size parameter B which will hold the requested # of elements.
	// For hint < 0 overLoadFactor returns false since hint < bucketCnt.
	// 理论上hint>8才会调用makemap函数, 所以B>=1
	// 获取map存储对数
	B := uint8(0)
	for overLoadFactor(hint, B) {//
		B++
	}
	h.B = B

	// allocate initial hash table
	// if B == 0, the buckets field is allocated lazily later (in mapassign)
	// If hint is large zeroing this memory could take a while.
	// if B==0, 则等到赋值的时候才会初始化buckets
	// if B!=0, 初始化buckets
	if h.B != 0 {
		var nextOverflow *bmap
		h.buckets, nextOverflow = makeBucketArray(t, h.B, nil)
		if nextOverflow != nil {// h.B<4, 则nextOverflow=nil
			h.extra = new(mapextra)
			h.extra.nextOverflow = nextOverflow
		}
	}

	return h
}

// makeBucketArray initializes a backing array for map buckets.
// 1<<b is the minimum number of buckets to allocate.
// dirtyalloc should either be nil or a bucket array previously
// allocated by makeBucketArray with the same t and b parameters.
// If dirtyalloc is nil a new backing array will be alloced and
// otherwise dirtyalloc will be cleared and reused as backing array.
func makeBucketArray(t *maptype, b uint8, dirtyalloc unsafe.Pointer) (buckets unsafe.Pointer, nextOverflow *bmap) {
	base := bucketShift(b)// 返回1<<b, 即2^b
	nbuckets := base

	// For small b, overflow buckets are unlikely.
	// Avoid the overhead of the calculation.
	if b >= 4 {
		// 桶对数>=4，需要额外申请多2^(b-4)个桶
		nbuckets += bucketShift(b - 4)
		sz := t.bucket.size * nbuckets
		up := roundupsize(sz)
		if up != sz {
			nbuckets = up / t.bucket.size
		}
	}

	if dirtyalloc == nil {
		// makemap中调用传入的dirtyalloc=nil
		// 初始化一个新的buckets
		buckets = newarray(t.bucket, int(nbuckets))
	} else {
		// dirtyalloc was previously generated by
		// the above newarray(t.bucket, int(nbuckets))
		// but may not be empty.
		buckets = dirtyalloc
		size := t.bucket.size * nbuckets
		if t.bucket.ptrdata != 0 {
			memclrHasPointers(buckets, size)
		} else {
			memclrNoHeapPointers(buckets, size)
		}
	}

	if base != nbuckets {// 当桶对数>=4，nbuckets会改变，此时进入该判断
		// We preallocated some overflow buckets.
		// To keep the overhead of tracking these overflow buckets to a minimum,
		// we use the convention that if a preallocated overflow bucket's overflow
		// pointer is nil, then there are more available by bumping the pointer.
		// We need a safe non-nil pointer for the last overflow bucket; just use buckets.
		nextOverflow = (*bmap)(add(buckets, base*uintptr(t.bucketsize)))
		last := (*bmap)(add(buckets, (nbuckets-1)*uintptr(t.bucketsize)))
		last.setoverflow(t, (*bmap)(buckets))
	}

	// b < 4, nbuckets值不变，nextOverflow=nil
	return buckets, nextOverflow
}

// mapaccess1 returns a pointer to h[key].  Never returns nil, instead
// it will return a reference to the zero object for the elem type if
// the key is not in the map.
// NOTE: The returned pointer may keep the whole map live, so don't
// hold onto it for very long.
// mapaccess1返回h[key]的指针，若找不到key则返回对应的零值
func mapaccess1(t *maptype, h *hmap, key unsafe.Pointer) unsafe.Pointer {
	if raceenabled && h != nil {// always false
		callerpc := getcallerpc()
		pc := funcPC(mapaccess1)
		racereadpc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	if msanenabled && h != nil {// always false
		msanread(key, t.key.size)
	}

	// h为空，或者元素为0，返回对应的零值
	if h == nil || h.count == 0 {
		if t.hashMightPanic() {
			t.key.alg.hash(key, 0) // see issue 23734
		}
		return unsafe.Pointer(&zeroVal[0])
	}

	// 读写冲突
	if h.flags&hashWriting != 0 {
		throw("concurrent map read and map write")
	}

	alg := t.key.alg// 拿到hash算法
	hash := alg.hash(key, uintptr(h.hash0))// 利用hash算法和种子计算hash值

	/**
	bucketMask: 1<<B-1
	当B=3, 1<<B-1 = 7, 转换为二进制=0111
	当B=4, 1<<B-1 = 15, 转换为二进制=1111
	作用：根据B的位数得到指定位数的1
	 */
	m := bucketMask(h.B)

	/*
	寻找buckets的过程：hash&m，即获取到hash的低m位，用来找到对应的桶的位置。
	比如64位hash：10010111 | 000011110110110010001111001010100010010110010101010 │ 01010
	假如b=5，则m=15，则buckets大小 = 2^5 = 32（0-31）
	二进制为1111（高位省略）, 1111 & hash = 01010，即桶在index=6的位置
	 */
	b := (*bmap)(add(h.buckets, (hash&m)*uintptr(t.bucketsize)))
	if c := h.oldbuckets; c != nil {
		if !h.sameSizeGrow() {
			// There used to be half as many buckets; mask down one more power of two.
			m >>= 1
		}
		oldb := (*bmap)(add(c, (hash&m)*uintptr(t.bucketsize)))
		if !evacuated(oldb) {// evacuated用于判断扩容是否已经完成
			b = oldb // 没有完成，遍历oldbuckets
		}
	}

	/**
	tophash用于取到hash的高8位，top用于定位key在buckets中的位置
	比如64位hash：10010111 | 000011110110110010001111001010100010010110010101010 │ 01010
	取到高8位即：10010111=151，则根据这个151去上一步骤的桶找到对应的tophash
	 */
	top := tophash(hash)
bucketloop:
	for ; b != nil; b = b.overflow(t) {// overflow返回下个桶的首地址
		for i := uintptr(0); i < bucketCnt; i++ {// 一个桶最多只能放8个元素
			if b.tophash[i] != top {// 遍历判断tophash中的值

				/*
				若遇到emptyRest，则直接退出循环
				这意味着后续的槽都是空的，overflow 指针也是空的
				 */
				if b.tophash[i] == emptyRest {
					break bucketloop
				}
				continue
			}
			// 获取key的偏移量，定位到key的位置
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			if t.indirectkey() {// key是指针，则解引用
				k = *((*unsafe.Pointer)(k))
			}
			if alg.equal(key, k) {// 若key相等
				// 定位到value的位置
				// bmap中key和value是分开存储的，即：key1/key2/key3/value1/value2/value3
				e := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.elemsize))
				if t.indirectelem() {// value是指针，则解引用
					e = *((*unsafe.Pointer)(e))
				}
				return e
			}
		}
	}
	// 遍历所有都没有找到，返回零值
	return unsafe.Pointer(&zeroVal[0])
}

func mapaccess2(t *maptype, h *hmap, key unsafe.Pointer) (unsafe.Pointer, bool) {
	if raceenabled && h != nil {
		callerpc := getcallerpc()
		pc := funcPC(mapaccess2)
		racereadpc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	if msanenabled && h != nil {
		msanread(key, t.key.size)
	}
	if h == nil || h.count == 0 {
		if t.hashMightPanic() {
			t.key.alg.hash(key, 0) // see issue 23734
		}
		return unsafe.Pointer(&zeroVal[0]), false
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map read and map write")
	}
	alg := t.key.alg
	hash := alg.hash(key, uintptr(h.hash0))
	m := bucketMask(h.B)
	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + (hash&m)*uintptr(t.bucketsize)))
	if c := h.oldbuckets; c != nil {
		// previous bucket array of half the size, non-nil only when growing
		// 当发生扩容时，oldbuckets不为空
		// 此时新的buckets还没有内容（未完成扩容），oldbuckets会存放扩容前的元素
		// 所以一定要在老得里面找，否则可能会发生消失的现象
		if !h.sameSizeGrow() {
			// 扩容时，m可能已经改变，若已经扩容，则只需要访问一半的buckets即可，所以m=m/2
			m >>= 1
		}
		// 获取oldbuckets的首地址
		oldb := (*bmap)(unsafe.Pointer(uintptr(c) + (hash&m)*uintptr(t.bucketsize)))
		if !evacuated(oldb) {
			b = oldb
		}
	}
	top := tophash(hash)
bucketloop:
	for ; b != nil; b = b.overflow(t) {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				if b.tophash[i] == emptyRest {
					break bucketloop
				}
				continue
			}
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			if t.indirectkey() {
				k = *((*unsafe.Pointer)(k))
			}
			if alg.equal(key, k) {
				e := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.elemsize))
				if t.indirectelem() {
					e = *((*unsafe.Pointer)(e))
				}
				return e, true
			}
		}
	}
	return unsafe.Pointer(&zeroVal[0]), false
}

// returns both key and elem. Used by map iterator
func mapaccessK(t *maptype, h *hmap, key unsafe.Pointer) (unsafe.Pointer, unsafe.Pointer) {
	if h == nil || h.count == 0 {
		return nil, nil
	}
	alg := t.key.alg
	hash := alg.hash(key, uintptr(h.hash0))
	m := bucketMask(h.B)
	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + (hash&m)*uintptr(t.bucketsize)))
	if c := h.oldbuckets; c != nil {
		if !h.sameSizeGrow() {
			// There used to be half as many buckets; mask down one more power of two.
			m >>= 1
		}
		oldb := (*bmap)(unsafe.Pointer(uintptr(c) + (hash&m)*uintptr(t.bucketsize)))
		if !evacuated(oldb) {
			b = oldb
		}
	}
	top := tophash(hash)
bucketloop:
	for ; b != nil; b = b.overflow(t) {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				if b.tophash[i] == emptyRest {
					break bucketloop
				}
				continue
			}
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			if t.indirectkey() {
				k = *((*unsafe.Pointer)(k))
			}
			if alg.equal(key, k) {
				e := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.elemsize))
				if t.indirectelem() {
					e = *((*unsafe.Pointer)(e))
				}
				return k, e
			}
		}
	}
	return nil, nil
}

func mapaccess1_fat(t *maptype, h *hmap, key, zero unsafe.Pointer) unsafe.Pointer {
	e := mapaccess1(t, h, key)
	if e == unsafe.Pointer(&zeroVal[0]) {
		return zero
	}
	return e
}

func mapaccess2_fat(t *maptype, h *hmap, key, zero unsafe.Pointer) (unsafe.Pointer, bool) {
	e := mapaccess1(t, h, key)
	if e == unsafe.Pointer(&zeroVal[0]) {
		return zero, false
	}
	return e, true
}

// Like mapaccess, but allocates a slot for the key if it is not present in the map.
// mapassign在runtime期间，会判断出key应该放到哪个地址，但不会将value放入
// 所以该函数没有传入value
func mapassign(t *maptype, h *hmap, key unsafe.Pointer) unsafe.Pointer {
	if h == nil {
		panic(plainError("assignment to entry in nil map"))
	}
	if raceenabled {
		callerpc := getcallerpc()
		pc := funcPC(mapassign)
		racewritepc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	if msanenabled {
		msanread(key, t.key.size)
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}

	// 获取对应类型的hash算法
	alg := t.key.alg
	hash := alg.hash(key, uintptr(h.hash0))

	// 因为alg.hash可能会造成panic，所以在调用alg.hash后需要将 hashWriting 的flag
	// 此时没有办法一次性完成写操作
	h.flags ^= hashWriting

	/*
	在makemap中，当h.B=0时，不会初始化buckets
	在进行set才会对buckets进行初始化操作，这叫做懒加载（用时才加载）
	 */
	if h.buckets == nil {
		h.buckets = newobject(t.bucket) // newarray(t.bucket, 1)
	}

again:
	// 获取低B位的值，用来定位bucket的位置
	bucket := hash & bucketMask(h.B)
	if h.growing() {// 正在扩容
		println("hmap is growing ", h.count)
		growWork(t, h, bucket)
	}

	// 获取对应bucket在内存中的首地址
	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + bucket*uintptr(t.bucketsize)))
	// 获取hash的高8位，用于定位tophash
	top := tophash(hash)

	var inserti *uint8         // 目标元素在桶中的索引
	var insertk unsafe.Pointer // key的地址
	var elem unsafe.Pointer    // value的地址
bucketloop:
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				/*
				在 b.tophash[i] != top 的情况下，可能会是一个空槽位 = emptyOne
				一般情况下 map 的槽位分布是这样的，e 表示 empty:
				[h1][h2][h3][h4][h5][e][e][e]
				但在执行过 delete 操作时，可能会变成这样:
				[h1][h2][e][e][h5][e][e][e]
				所以如果再插入的话，会尽量往前面的位置插
				[h1][h2][e][e][h5][e][e][e]
				即执行插入操作后，可能变为
				[h1][h2][h6][e][h5][e][e][e]
				 */
				if isEmpty(b.tophash[i]) && inserti == nil {
					// 该槽没有被占，可以向其中插入元素
					inserti = &b.tophash[i] // tophash插入的位置
					insertk = add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
					elem = add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.elemsize))
				}
				if b.tophash[i] == emptyRest {// 不用找了，现有的buckets中没有槽位供插入了
					break bucketloop
				}
				continue
			}

			// 找到了一个槽位
			// 根据i计算k的地址（首地址+偏移量）
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			if t.indirectkey() {
				k = *((*unsafe.Pointer)(k))
			}

			// key不相等，但是落到同一个桶，说明发生了哈希碰撞
			if !alg.equal(key, k) {
				continue
			}

			/**
			具体issue：https://github.com/golang/go/issues/11088
			解决issue的pr：https://github.com/mcrwayfun/go/commit/00c638d243056b24f1deeb2d1d954e62baedd468

			为什么需要进行key的覆盖？理论上key是不需要覆盖的？
			比如：当map中已经存在key=0(int), 当下个key=0进行set操作时，直接进行value的覆盖操作不就好了吗？
			当程序走到这里，说明已经找到了槽位，也解决了哈希碰撞，理论上两个key应该是一样的。

			实际上，key type有区分为 "well-behaved" (the type's equality is byte-by-byte equality) 和其他，
			当key type是well-behaved时，则可以不用更新key，而是否为well-behaved，由方法needkeyupdate决定。
			举个例子：对于 m[-0]和m[+0]会被落到同一个槽，但是key确是不一样的，所以需要覆盖！
			 */
			if t.needkeyupdate() {
				typedmemmove(t.key, k, key)
			}

			// 获取到应该插入的value的地址
			elem = add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.elemsize))
			goto done
		}

		// bucket的8个槽没有找到能够插入值或者更新值的，则遍历overflow去下个bucket找
		ovf := b.overflow(t)
		if ovf == nil {// overflow已经为空了
			break
		}
		// 赋值为链表的下一个元素，继续循环
		b = ovf
	}

	// 没有找到对应的key，则分配新的空间

	// 如果已经超出了最大的 load factor，或者已经有太多的 overflow buckets
	// 并且这个时刻没有在运行 growing 的途中，则开始进行 growing
	// 满足以下3个条件会进入扩容：
	// 1：没有在扩容
	// 2：装在因子超过了6.5
	// 3：hash表使用了太多的溢出桶
	if !h.growing() && (overLoadFactor(h.count+1, h.B) || tooManyOverflowBuckets(h.noverflow, h.B)) {
		hashGrow(t, h)
		goto again // Growing the table invalidates everything, so try again
	}

	if inserti == nil {
		// all current buckets are full, allocate a new one.
		// 之前在桶中找不到能存入这个tophash的位置
		// 说明当前所有的 buckets 都是满的，会分配一个新的 bucket

		newb := h.newoverflow(t, b) // 创建一个新的map，并将之前的overflow连上新的节点
		inserti = &newb.tophash[0]  // 设置tophash的值
		insertk = add(unsafe.Pointer(newb), dataOffset)
		elem = add(insertk, bucketCnt*uintptr(t.keysize))
	}

	// store new key/elem at insert position

	if t.indirectkey() {// key是指针
		kmem := newobject(t.key)
		*(*unsafe.Pointer)(insertk) = kmem
		insertk = kmem // 解引用并拿到值
	}
	if t.indirectelem() {// value是指针，
		vmem := newobject(t.elem)
		*(*unsafe.Pointer)(elem) = vmem // 解引用
	}
	typedmemmove(t.key, insertk, key) // 将新的key存储到对应的位置
	*inserti = top                    // tophash 插入指定的位置
	h.count++                         // map元素加1

done:
	if h.flags&hashWriting == 0 {
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting
	if t.indirectelem() {
		elem = *((*unsafe.Pointer)(elem))
	}
	return elem
}

// map删除操作
func mapdelete(t *maptype, h *hmap, key unsafe.Pointer) {
	if raceenabled && h != nil {
		callerpc := getcallerpc()
		pc := funcPC(mapdelete)
		racewritepc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	if msanenabled && h != nil {
		msanread(key, t.key.size)
	}
	if h == nil || h.count == 0 {
		if t.hashMightPanic() {
			t.key.alg.hash(key, 0) // see issue 23734
		}
		return
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}

	alg := t.key.alg
	hash := alg.hash(key, uintptr(h.hash0))

	// 因为alg.hash可能会造成panic，所以在调用alg.hash后需要将 hashWriting 的flag
	// 此时没有办法一次性完成删除操作
	h.flags ^= hashWriting

	bucket := hash & bucketMask(h.B)
	if h.growing() {
		growWork(t, h, bucket)
	}
	b := (*bmap)(add(h.buckets, bucket*uintptr(t.bucketsize)))
	bOrig := b// 保存现有的bucket
	top := tophash(hash)
search:
	for ; b != nil; b = b.overflow(t) {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				if b.tophash[i] == emptyRest {// 跳出search
					break search
				}
				continue
			}

			// tophash == top
			// 找到key的首地址
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			// 保存k的地址
			k2 := k
			if t.indirectkey() {// 解引用
				k2 = *((*unsafe.Pointer)(k2))
			}

			// key和k2的hash不想等，可能是有hash碰撞，继续遍历
			if !alg.equal(key, k2) {
				continue
			}
			// Only clear key if there are pointers in it.
			// 当key是指针时，才清除key
			if t.indirectkey() {
				*(*unsafe.Pointer)(k) = nil
			} else if t.key.ptrdata != 0 {
				memclrHasPointers(k, t.key.size)
			}

			// 获取value的地址
			e := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.elemsize))
			if t.indirectelem() {// value如果是指针的话，则将value设置为空
				*(*unsafe.Pointer)(e) = nil
			} else if t.elem.ptrdata != 0 {
				memclrHasPointers(e, t.elem.size)
			} else {
				memclrNoHeapPointers(e, t.elem.size)
			}
			// 设置key对应的tophash为 emptyOne
			b.tophash[i] = emptyOne
			// If the bucket now ends in a bunch of emptyOne states,
			// change those to emptyRest states.
			// It would be nice to make this a separate function, but
			// for loops are not currently inlineable.

			/**
			如果bucket以一堆emptyRest结束，则将它们修改为emptyRest状态，否则跳转到notLast
			举个例子，分两种情况：
			1：被删除槽位于bmap的尾部，即i=bucketCnt-1
			[1][1][x][1][x][x][x][emptyOne]
			if 后继bucket的bmap的tophash[0]是emptyRest，
				从当前槽位向前，将它们的tophash修改为emptyRest，直到遇到非空槽位则停止
				[1][1][x][1][emptyRest][emptyRest][emptyRest][emptyRest]
			else
				goto ontLast

			2：被删除槽位不在bmap尾部，
			[1][1][x][1][x][x][emptyOne][x]
			if 当前槽位的下个槽位是emptyRest，
				从当前槽位向前，将它们的tophash修改为emptyRest，直到遇到非空槽位则停止
				[1][1][x][1][emptyRest][emptyRest][emptyRest][emptyRest]
			else
				goto ontLast
			 */
			if i == bucketCnt-1 {
				if b.overflow(t) != nil && b.overflow(t).tophash[0] != emptyRest {
					goto notLast
				}
			} else {
				if b.tophash[i+1] != emptyRest {
					goto notLast
				}
			}


			// 从当前槽位向前，将它们的tophash修改为emptyRest，直到遇到非空槽位则停止
			for {
				// 设置当前tophash为 emptyRest
				b.tophash[i] = emptyRest
				if i == 0 {
					if b == bOrig {
						break // b为进入mapdelete后定位的第一个桶，则break该for循环
					}
					c := b
					// 找到bucket中 bmap == c的 bmap（？？？这里是什么操作？）
					for b = bOrig; b.overflow(t) != c; b = b.overflow(t) {
					}
					i = bucketCnt - 1
				} else {
					i--
				}

				// 当前bmap != emptyOne，则break该for循环
				if b.tophash[i] != emptyOne {
					break
				}
			}
		notLast:
			h.count-- // 元素减1
			break search
		}
	}

	if h.flags&hashWriting == 0 {// 不允许同时读写（map delete时不允许读）
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting
}

// mapiterinit initializes the hiter struct used for ranging over maps.
// The hiter struct pointed to by 'it' is allocated on the stack
// by the compilers order pass or on the heap by reflect_mapiterinit.
// Both need to have zeroed hiter since the struct contains pointers.
func mapiterinit(t *maptype, h *hmap, it *hiter) {
	if raceenabled && h != nil {
		callerpc := getcallerpc()
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(mapiterinit))
	}

	if h == nil || h.count == 0 {
		return
	}

	if unsafe.Sizeof(hiter{})/sys.PtrSize != 12 {
		throw("hash_iter size incorrect") // see cmd/compile/internal/gc/reflect.go
	}
	it.t = t
	it.h = h

	// grab snapshot of bucket state
	it.B = h.B
	it.buckets = h.buckets
	if t.bucket.ptrdata == 0 {
		// Allocate the current slice and remember pointers to both current and old.
		// This preserves all relevant overflow buckets alive even if
		// the table grows and/or overflow buckets are added to the table
		// while we are iterating.
		h.createOverflow()
		it.overflow = h.extra.overflow
		it.oldoverflow = h.extra.oldoverflow
	}

	// decide where to start
	r := uintptr(fastrand())
	if h.B > 31-bucketCntBits {
		r += uintptr(fastrand()) << 31
	}
	it.startBucket = r & bucketMask(h.B)
	it.offset = uint8(r >> h.B & (bucketCnt - 1))

	// iterator state
	it.bucket = it.startBucket

	// Remember we have an iterator.
	// Can run concurrently with another mapiterinit().
	if old := h.flags; old&(iterator|oldIterator) != iterator|oldIterator {
		atomic.Or8(&h.flags, iterator|oldIterator)
	}

	mapiternext(it)
}

func mapiternext(it *hiter) {
	h := it.h
	if raceenabled {
		callerpc := getcallerpc()
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(mapiternext))
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map iteration and map write")
	}
	t := it.t
	bucket := it.bucket
	b := it.bptr
	i := it.i
	checkBucket := it.checkBucket
	alg := t.key.alg

next:
	if b == nil {
		if bucket == it.startBucket && it.wrapped {
			// end of iteration
			it.key = nil
			it.elem = nil
			return
		}
		if h.growing() && it.B == h.B {
			// Iterator was started in the middle of a grow, and the grow isn't done yet.
			// If the bucket we're looking at hasn't been filled in yet (i.e. the old
			// bucket hasn't been evacuated) then we need to iterate through the old
			// bucket and only return the ones that will be migrated to this bucket.
			oldbucket := bucket & it.h.oldbucketmask()
			b = (*bmap)(add(h.oldbuckets, oldbucket*uintptr(t.bucketsize)))
			if !evacuated(b) {
				checkBucket = bucket
			} else {
				b = (*bmap)(add(it.buckets, bucket*uintptr(t.bucketsize)))
				checkBucket = noCheck
			}
		} else {
			b = (*bmap)(add(it.buckets, bucket*uintptr(t.bucketsize)))
			checkBucket = noCheck
		}
		bucket++
		if bucket == bucketShift(it.B) {
			bucket = 0
			it.wrapped = true
		}
		i = 0
	}
	for ; i < bucketCnt; i++ {
		offi := (i + it.offset) & (bucketCnt - 1)
		if isEmpty(b.tophash[offi]) || b.tophash[offi] == evacuatedEmpty {
			// TODO: emptyRest is hard to use here, as we start iterating
			// in the middle of a bucket. It's feasible, just tricky.
			continue
		}
		k := add(unsafe.Pointer(b), dataOffset+uintptr(offi)*uintptr(t.keysize))
		if t.indirectkey() {
			k = *((*unsafe.Pointer)(k))
		}
		e := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+uintptr(offi)*uintptr(t.elemsize))
		if checkBucket != noCheck && !h.sameSizeGrow() {
			// Special case: iterator was started during a grow to a larger size
			// and the grow is not done yet. We're working on a bucket whose
			// oldbucket has not been evacuated yet. Or at least, it wasn't
			// evacuated when we started the bucket. So we're iterating
			// through the oldbucket, skipping any keys that will go
			// to the other new bucket (each oldbucket expands to two
			// buckets during a grow).
			if t.reflexivekey() || alg.equal(k, k) {
				// If the item in the oldbucket is not destined for
				// the current new bucket in the iteration, skip it.
				hash := alg.hash(k, uintptr(h.hash0))
				if hash&bucketMask(it.B) != checkBucket {
					continue
				}
			} else {
				// Hash isn't repeatable if k != k (NaNs).  We need a
				// repeatable and randomish choice of which direction
				// to send NaNs during evacuation. We'll use the low
				// bit of tophash to decide which way NaNs go.
				// NOTE: this case is why we need two evacuate tophash
				// values, evacuatedX and evacuatedY, that differ in
				// their low bit.
				if checkBucket>>(it.B-1) != uintptr(b.tophash[offi]&1) {
					continue
				}
			}
		}
		if (b.tophash[offi] != evacuatedX && b.tophash[offi] != evacuatedY) ||
			!(t.reflexivekey() || alg.equal(k, k)) {
			// This is the golden data, we can return it.
			// OR
			// key!=key, so the entry can't be deleted or updated, so we can just return it.
			// That's lucky for us because when key!=key we can't look it up successfully.
			it.key = k
			if t.indirectelem() {
				e = *((*unsafe.Pointer)(e))
			}
			it.elem = e
		} else {
			// The hash table has grown since the iterator was started.
			// The golden data for this key is now somewhere else.
			// Check the current hash table for the data.
			// This code handles the case where the key
			// has been deleted, updated, or deleted and reinserted.
			// NOTE: we need to regrab the key as it has potentially been
			// updated to an equal() but not identical key (e.g. +0.0 vs -0.0).
			rk, re := mapaccessK(t, h, k)
			if rk == nil {
				continue // key has been deleted
			}
			it.key = rk
			it.elem = re
		}
		it.bucket = bucket
		if it.bptr != b { // avoid unnecessary write barrier; see issue 14921
			it.bptr = b
		}
		it.i = i + 1
		it.checkBucket = checkBucket
		return
	}
	b = b.overflow(t)
	i = 0
	goto next
}

// mapclear deletes all keys from a map.
func mapclear(t *maptype, h *hmap) {
	if raceenabled && h != nil {
		callerpc := getcallerpc()
		pc := funcPC(mapclear)
		racewritepc(unsafe.Pointer(h), callerpc, pc)
	}

	if h == nil || h.count == 0 {
		return
	}

	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}

	h.flags ^= hashWriting

	h.flags &^= sameSizeGrow
	h.oldbuckets = nil
	h.nevacuate = 0
	h.noverflow = 0
	h.count = 0

	// Keep the mapextra allocation but clear any extra information.
	if h.extra != nil {
		*h.extra = mapextra{}
	}

	// makeBucketArray clears the memory pointed to by h.buckets
	// and recovers any overflow buckets by generating them
	// as if h.buckets was newly alloced.
	_, nextOverflow := makeBucketArray(t, h.B, h.buckets)
	if nextOverflow != nil {
		// If overflow buckets are created then h.extra
		// will have been allocated during initial bucket creation.
		h.extra.nextOverflow = nextOverflow
	}

	if h.flags&hashWriting == 0 {
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting
}

func hashGrow(t *maptype, h *hmap) {
	// 1：装在因子超过了6.5
	// 2：hash表使用了太多的溢出桶
	// if case 1：需要对map进行扩容，即B=B+1，bucket总数会变为原来的两倍
	// if case 2：则保持当前bucket的数量不变，flag设置为sameSizeGrow，横向拓展？
	bigger := uint8(1)
	if !overLoadFactor(h.count+1, h.B) {
		bigger = 0
		h.flags |= sameSizeGrow
	}
	oldbuckets := h.buckets
	newbuckets, nextOverflow := makeBucketArray(t, h.B+bigger, nil)

	// 先把flags中的iterator和oldlterator 对应的位清0
	// buckets现在已经挂到了oldbuckets的名下了，对应的标志位也要转移过去
	flags := h.flags &^ (iterator | oldIterator)
	if h.flags&iterator != 0 {
		flags |= oldIterator
	}
	// 提交扩容结果
	h.B += bigger
	h.flags = flags
	h.oldbuckets = oldbuckets // 旧的buckets放到 oldbuckets这个指针上
	h.buckets = newbuckets    // 新的buckets放到 buckets
	h.nevacuate = 0
	h.noverflow = 0

	if h.extra != nil && h.extra.overflow != nil {
		// 把当前的 overflow 赋值给 old overflow
		if h.extra.oldoverflow != nil {
			throw("oldoverflow is not nil")
		}
		h.extra.oldoverflow = h.extra.overflow
		h.extra.overflow = nil
	}
	if nextOverflow != nil {
		if h.extra == nil {
			h.extra = new(mapextra)
		}
		// 存储over flow空闲空间
		// src/runtime/map.go:770
		h.extra.nextOverflow = nextOverflow
	}

	// the actual copying of the hash table data is done incrementally
	// by growWork() and evacuate().
	// 实际的哈希表元素的拷贝工作是在 growWork 和 evacuate 中增量慢慢地进行的
}

// overLoadFactor reports whether count items placed in 1<<B buckets is over loadFactor.
func overLoadFactor(count int, B uint8) bool {
	return count > bucketCnt && uintptr(count) > loadFactorNum*(bucketShift(B)/loadFactorDen)
}

// tooManyOverflowBuckets reports whether noverflow buckets is too many for a map with 1<<B buckets.
// Note that most of these overflow buckets must be in sparse use;
// if use was dense, then we'd have already triggered regular map growth.
func tooManyOverflowBuckets(noverflow uint16, B uint8) bool {
	// If the threshold is too low, we do extraneous work.
	// If the threshold is too high, maps that grow and shrink can hold on to lots of unused memory.
	// "too many" means (approximately) as many overflow buckets as regular buckets.
	// See incrnoverflow for more details.
	if B > 15 {
		B = 15
	}
	// The compiler doesn't see here that B < 16; mask B to generate shorter shift code.
	return noverflow >= uint16(1)<<(B&15)
}

// growing reports whether h is growing. The growth may be to the same size or bigger.
func (h *hmap) growing() bool {
	return h.oldbuckets != nil
}

// sameSizeGrow reports whether the current growth is to a map of the same size.
func (h *hmap) sameSizeGrow() bool {
	return h.flags&sameSizeGrow != 0
}

// noldbuckets calculates the number of buckets prior to the current map growth.
func (h *hmap) noldbuckets() uintptr {
	oldB := h.B
	if !h.sameSizeGrow() {
		oldB--
	}
	return bucketShift(oldB)
}

// oldbucketmask provides a mask that can be applied to calculate n % noldbuckets().
func (h *hmap) oldbucketmask() uintptr {
	return h.noldbuckets() - 1
}

func growWork(t *maptype, h *hmap, bucket uintptr) {
	// 确保我们搬迁的 oldbucket对应的是我们马上就要使用的那一个
	// 之前使用key&hash的低五位来确定bucket的位置
	evacuate(t, h, bucket&h.oldbucketmask())

	// evacuate one more oldbucket to make progress on growing
	// 如果是 growing 状态，再多移一个 oldbucket
	if h.growing() {
		evacuate(t, h, h.nevacuate)
	}
}

func bucketEvacuated(t *maptype, h *hmap, bucket uintptr) bool {
	b := (*bmap)(add(h.oldbuckets, bucket*uintptr(t.bucketsize)))
	return evacuated(b)
}

// evacDst is an evacuation destination.
type evacDst struct {
	b *bmap          // current destination bucket
	i int            // key/elem index into b
	k unsafe.Pointer // pointer to current key storage
	e unsafe.Pointer // pointer to current elem storage
}

func evacuate(t *maptype, h *hmap, oldbucket uintptr) {
	// 定位oldbucket的位置
	b := (*bmap)(add(h.oldbuckets, oldbucket*uintptr(t.bucketsize)))
	// 获取oldbucket的数量
	newbit := h.noldbuckets()
	// 如果b没有被搬迁过
	if !evacuated(b) {
		// TODO: reuse overflow buckets instead of using new ones, if there
		// is no iterator using the old buckets.  (If !oldIterator.)

		// xy contains the x and y (low and high) evacuation destinations.
		// xy包含移动的目标
		// x表示bucket的前半部分，y表示bucket的后半部分
		var xy [2]evacDst
		x := &xy[0]
		x.b = (*bmap)(add(h.buckets, oldbucket*uintptr(t.bucketsize)))
		x.k = add(unsafe.Pointer(x.b), dataOffset)
		x.e = add(x.k, bucketCnt*uintptr(t.keysize))

		if !h.sameSizeGrow() {
			// Only calculate y pointers if we're growing bigger.
			// Otherwise GC can see bad pointers.
			y := &xy[1]
			y.b = (*bmap)(add(h.buckets, (oldbucket+newbit)*uintptr(t.bucketsize)))
			y.k = add(unsafe.Pointer(y.b), dataOffset)
			y.e = add(y.k, bucketCnt*uintptr(t.keysize))
		}

		// 遍历所有的bucket，包括overflow的
		for ; b != nil; b = b.overflow(t) {
			k := add(unsafe.Pointer(b), dataOffset)
			e := add(k, bucketCnt*uintptr(t.keysize))
			// 遍历bucket中所有的槽位
			for i := 0; i < bucketCnt; i, k, e = i+1, add(k, uintptr(t.keysize)), add(e, uintptr(t.elemsize)) {
				// 当前槽位的tophash
				top := b.tophash[i]
				// 如果当前槽位为空，则标记为已经搬迁过，并继续下一个槽位的操作
				if isEmpty(top) {
					b.tophash[i] = evacuatedEmpty
					continue
				}
				if top < minTopHash {
					throw("bad map state")
				}
				k2 := k
				if t.indirectkey() {// 解引用
					k2 = *((*unsafe.Pointer)(k2))
				}
				var useY uint8
				if !h.sameSizeGrow() {
					// 计算hash，以判断我们的数据要转移到哪一部分的bucket
					// 可能是x部分，也可能是y部分
					hash := t.key.alg.hash(k2, uintptr(h.hash0))
					if h.flags&iterator != 0 && !t.reflexivekey() && !t.key.alg.equal(k2, k2) {
						// If key != key (NaNs), then the hash could be (and probably
						// will be) entirely different from the old hash. Moreover,
						// it isn't reproducible. Reproducibility is required in the
						// presence of iterators, as our evacuation decision must
						// match whatever decision the iterator made.
						// Fortunately, we have the freedom to send these keys either
						// way. Also, tophash is meaningless for these kinds of keys.
						// We let the low bit of tophash drive the evacuation decision.
						// We recompute a new random tophash for the next level so
						// these keys will get evenly distributed across all buckets
						// after multiple grows.
						// 为什么要加 reflexivekey 的判断，可以参考这里:
						// https://go-review.googlesource.com/c/go/+/1480
						// key != key，只有在 float 数的 NaN 时会出现
						// 比如:
						// n1 := math.NaN()
						// n2 := math.NaN()
						// fmt.Println(n1, n2)
						// fmt.Println(n1 == n2)
						// 这种情况下 n1 和 n2 的哈希值也完全不一样
						// 这里官方表示这种情况是不可复现的
						// 需要在 iterators 参与的情况下才能复现
						// 但是对于这种 key 我们也可以随意对其目标进行发配
						// 同时 tophash 对于 NaN 也没啥意义
						// 还是按正常的情况下算一个随机的 tophash
						// 然后公平地把这些 key 平均分布到各 bucket 就好
						useY = top & 1
						top = tophash(hash)
					} else {
						// 数据迁移到y部分
						if hash&newbit != 0 {
							// println("hash&newbit != 0, hash:%v, newbit:%v, hash&newbit:%v", hash, newbit, hash&newbit)
							useY = 1
						}
					}
				}

				if evacuatedX+1 != evacuatedY || evacuatedX^1 != evacuatedY {
					throw("bad evacuatedN")
				}

				b.tophash[i] = evacuatedX + useY // evacuatedX + 1 == evacuatedY
				dst := &xy[useY]                 // 移动目标

				if dst.i == bucketCnt {
					dst.b = h.newoverflow(t, dst.b)
					dst.i = 0
					dst.k = add(unsafe.Pointer(dst.b), dataOffset)
					dst.e = add(dst.k, bucketCnt*uintptr(t.keysize))
				}
				dst.b.tophash[dst.i&(bucketCnt-1)] = top // mask dst.i as an optimization, to avoid a bounds check
				if t.indirectkey() {// 拷贝指针
					*(*unsafe.Pointer)(dst.k) = k2 // copy pointer
				} else {
					typedmemmove(t.key, dst.k, k) // 拷贝值
				}
				if t.indirectelem() {
					*(*unsafe.Pointer)(dst.e) = *(*unsafe.Pointer)(e)
				} else {
					typedmemmove(t.elem, dst.e, e)
				}
				dst.i++
				// These updates might push these pointers past the end of the
				// key or elem arrays.  That's ok, as we have the overflow pointer
				// at the end of the bucket to protect against pointing past the
				// end of the bucket.
				dst.k = add(dst.k, uintptr(t.keysize))
				dst.e = add(dst.e, uintptr(t.elemsize))
			}
		}
		// Unlink the overflow buckets & clear key/elem to help GC.
		if h.flags&oldIterator == 0 && t.bucket.ptrdata != 0 {
			b := add(h.oldbuckets, oldbucket*uintptr(t.bucketsize))
			// Preserve b.tophash because the evacuation
			// state is maintained there.
			ptr := add(b, dataOffset)
			n := uintptr(t.bucketsize) - dataOffset
			memclrHasPointers(ptr, n)
		}
	}

	// 如果此次搬迁的bucket等于当前进度，则更新搬迁进度
	if oldbucket == h.nevacuate {
		advanceEvacuationMark(h, t, newbit)
	}
}

func advanceEvacuationMark(h *hmap, t *maptype, newbit uintptr) {
	// 进度+1
	h.nevacuate++
	// Experiments suggest that 1024 is overkill by at least an order of magnitude.
	// Put it in there as a safeguard anyway, to ensure O(1) behav
	// 尝试向后看1024 bucket
	stop := h.nevacuate + 1024
	if stop > newbit {
		stop = newbit
	}
	// 寻找没有搬迁的bucekt
	for h.nevacuate != stop && bucketEvacuated(t, h, h.nevacuate) {
		h.nevacuate++
	}

	// 所有bucket已经搬迁完成
	if h.nevacuate == newbit { // newbit == # of oldbuckets
		// Growing is all done. Free old main bucket array.
		h.oldbuckets = nil
		// Can discard old overflow buckets as well.
		// If they are still referenced by an iterator,
		// then the iterator holds a pointers to the slice.
		if h.extra != nil {
			h.extra.oldoverflow = nil
		}
		h.flags &^= sameSizeGrow
	}
}

func ismapkey(t *_type) bool {
	return t.alg.hash != nil
}

// Reflect stubs. Called from ../reflect/asm_*.s

//go:linkname reflect_makemap reflect.makemap
func reflect_makemap(t *maptype, cap int) *hmap {
	// Check invariants and reflects math.
	if !ismapkey(t.key) {
		throw("runtime.reflect_makemap: unsupported map key type")
	}
	if t.key.size > maxKeySize && (!t.indirectkey() || t.keysize != uint8(sys.PtrSize)) ||
		t.key.size <= maxKeySize && (t.indirectkey() || t.keysize != uint8(t.key.size)) {
		throw("key size wrong")
	}
	if t.elem.size > maxElemSize && (!t.indirectelem() || t.elemsize != uint8(sys.PtrSize)) ||
		t.elem.size <= maxElemSize && (t.indirectelem() || t.elemsize != uint8(t.elem.size)) {
		throw("elem size wrong")
	}
	if t.key.align > bucketCnt {
		throw("key align too big")
	}
	if t.elem.align > bucketCnt {
		throw("elem align too big")
	}
	if t.key.size%uintptr(t.key.align) != 0 {
		throw("key size not a multiple of key align")
	}
	if t.elem.size%uintptr(t.elem.align) != 0 {
		throw("elem size not a multiple of elem align")
	}
	if bucketCnt < 8 {
		throw("bucketsize too small for proper alignment")
	}
	if dataOffset%uintptr(t.key.align) != 0 {
		throw("need padding in bucket (key)")
	}
	if dataOffset%uintptr(t.elem.align) != 0 {
		throw("need padding in bucket (elem)")
	}

	return makemap(t, cap, nil)
}

//go:linkname reflect_mapaccess reflect.mapaccess
func reflect_mapaccess(t *maptype, h *hmap, key unsafe.Pointer) unsafe.Pointer {
	elem, ok := mapaccess2(t, h, key)
	if !ok {
		// reflect wants nil for a missing element
		elem = nil
	}
	return elem
}

//go:linkname reflect_mapassign reflect.mapassign
func reflect_mapassign(t *maptype, h *hmap, key unsafe.Pointer, elem unsafe.Pointer) {
	p := mapassign(t, h, key)
	typedmemmove(t.elem, p, elem)
}

//go:linkname reflect_mapdelete reflect.mapdelete
func reflect_mapdelete(t *maptype, h *hmap, key unsafe.Pointer) {
	mapdelete(t, h, key)
}

//go:linkname reflect_mapiterinit reflect.mapiterinit
func reflect_mapiterinit(t *maptype, h *hmap) *hiter {
	it := new(hiter)
	mapiterinit(t, h, it)
	return it
}

//go:linkname reflect_mapiternext reflect.mapiternext
func reflect_mapiternext(it *hiter) {
	mapiternext(it)
}

//go:linkname reflect_mapiterkey reflect.mapiterkey
func reflect_mapiterkey(it *hiter) unsafe.Pointer {
	return it.key
}

//go:linkname reflect_mapiterelem reflect.mapiterelem
func reflect_mapiterelem(it *hiter) unsafe.Pointer {
	return it.elem
}

//go:linkname reflect_maplen reflect.maplen
func reflect_maplen(h *hmap) int {
	if h == nil {
		return 0
	}
	if raceenabled {
		callerpc := getcallerpc()
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(reflect_maplen))
	}
	return h.count
}

//go:linkname reflectlite_maplen internal/reflectlite.maplen
func reflectlite_maplen(h *hmap) int {
	if h == nil {
		return 0
	}
	if raceenabled {
		callerpc := getcallerpc()
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(reflect_maplen))
	}
	return h.count
}

//go:linkname reflect_ismapkey reflect.ismapkey
func reflect_ismapkey(t *_type) bool {
	return ismapkey(t)
}

const maxZero = 1024 // must match value in cmd/compile/internal/gc/walk.go
var zeroVal [maxZero]byte
