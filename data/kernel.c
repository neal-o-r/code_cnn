int acct_parm[3] = {4, 2, 30};





struct bsd_acct_struct {
 struct fs_pin pin;
 atomic_long_t count;
 struct rcu_head rcu;
 struct mutex lock;
 int active;
 unsigned long needcheck;
 struct file *file;
 struct pid_namespace *ns;
 struct work_struct work;
 struct completion done;
};

static void do_acct_process(struct bsd_acct_struct *acct);




static int check_free_space(struct bsd_acct_struct *acct)
{
 struct kstatfs sbuf;

 if (time_is_before_jiffies(acct->needcheck))
  goto out;


 if (vfs_statfs(&acct->file->f_path, &sbuf))
  goto out;

 if (acct->active) {
  u64 suspend = sbuf.f_blocks * SUSPEND;
  do_div(suspend, 100);
  if (sbuf.f_bavail <= suspend) {
   acct->active = 0;
   pr_info("Process accounting paused\n");
  }
 } else {
  u64 resume = sbuf.f_blocks * RESUME;
  do_div(resume, 100);
  if (sbuf.f_bavail >= resume) {
   acct->active = 1;
   pr_info("Process accounting resumed\n");
  }
 }

 acct->needcheck = jiffies + ACCT_TIMEOUT*HZ;
 out:
 return acct->active;
}

static void acct_put(struct bsd_acct_struct *p)
{
 if (atomic_long_dec_and_test(&p->count))
  kfree_rcu(p, rcu);
}

static inline struct bsd_acct_struct *to_acct(struct fs_pin *p)
{
 return p ? container_of(p, struct bsd_acct_struct, pin) : NULL;
}

static struct bsd_acct_struct *acct_get(struct pid_namespace *ns)
{
 struct bsd_acct_struct *res;
 again:
 smp_rmb();
 rcu_read_lock();
 res = to_acct(ACCESS_ONCE(ns->bacct));
 if (!res) {
  rcu_read_unlock();
  return NULL;
 }
 if (!atomic_long_inc_not_zero(&res->count)) {
  rcu_read_unlock();
  cpu_relax();
  goto again;
 }
 rcu_read_unlock();
 mutex_lock(&res->lock);
 if (res != to_acct(ACCESS_ONCE(ns->bacct))) {
  mutex_unlock(&res->lock);
  acct_put(res);
  goto again;
 }
 return res;
}

static void acct_pin_kill(struct fs_pin *pin)
{
 struct bsd_acct_struct *acct = to_acct(pin);
 mutex_lock(&acct->lock);
 do_acct_process(acct);
 schedule_work(&acct->work);
 wait_for_completion(&acct->done);
 cmpxchg(&acct->ns->bacct, pin, NULL);
 mutex_unlock(&acct->lock);
 pin_remove(pin);
 acct_put(acct);
}

static void close_work(struct work_struct *work)
{
 struct bsd_acct_struct *acct = container_of(work, struct bsd_acct_struct, work);
 struct file *file = acct->file;
 if (file->f_op->flush)
  file->f_op->flush(file, NULL);
 __fput_sync(file);
 complete(&acct->done);
}

static int acct_on(struct filename *pathname)
{
 struct file *file;
 struct vfsmount *mnt, *internal;
 struct pid_namespace *ns = task_active_pid_ns(current);
 struct bsd_acct_struct *acct;
 struct fs_pin *old;
 int err;

 acct = kzalloc(sizeof(struct bsd_acct_struct), GFP_KERNEL);
 if (!acct)
  return -ENOMEM;


 file = file_open_name(pathname, O_WRONLY|O_APPEND|O_LARGEFILE, 0);
 if (IS_ERR(file)) {
  kfree(acct);
  return PTR_ERR(file);
 }

 if (!S_ISREG(file_inode(file)->i_mode)) {
  kfree(acct);
  filp_close(file, NULL);
  return -EACCES;
 }

 if (!(file->f_mode & FMODE_CAN_WRITE)) {
  kfree(acct);
  filp_close(file, NULL);
  return -EIO;
 }
 internal = mnt_clone_internal(&file->f_path);
 if (IS_ERR(internal)) {
  kfree(acct);
  filp_close(file, NULL);
  return PTR_ERR(internal);
 }
 err = mnt_want_write(internal);
 if (err) {
  mntput(internal);
  kfree(acct);
  filp_close(file, NULL);
  return err;
 }
 mnt = file->f_path.mnt;
 file->f_path.mnt = internal;

 atomic_long_set(&acct->count, 1);
 init_fs_pin(&acct->pin, acct_pin_kill);
 acct->file = file;
 acct->needcheck = jiffies;
 acct->ns = ns;
 mutex_init(&acct->lock);
 INIT_WORK(&acct->work, close_work);
 init_completion(&acct->done);
 mutex_lock_nested(&acct->lock, 1);
 pin_insert(&acct->pin, mnt);

 rcu_read_lock();
 old = xchg(&ns->bacct, &acct->pin);
 mutex_unlock(&acct->lock);
 pin_kill(old);
 mnt_drop_write(mnt);
 mntput(mnt);
 return 0;
}

static DEFINE_MUTEX(acct_on_mutex);
SYSCALL_DEFINE1(acct, const char __user *, name)
{
 int error = 0;

 if (!capable(CAP_SYS_PACCT))
  return -EPERM;

 if (name) {
  struct filename *tmp = getname(name);

  if (IS_ERR(tmp))
   return PTR_ERR(tmp);
  mutex_lock(&acct_on_mutex);
  error = acct_on(tmp);
  mutex_unlock(&acct_on_mutex);
  putname(tmp);
 } else {
  rcu_read_lock();
  pin_kill(task_active_pid_ns(current)->bacct);
 }

 return error;
}

void acct_exit_ns(struct pid_namespace *ns)
{
 rcu_read_lock();
 pin_kill(ns->bacct);
}

static comp_t encode_comp_t(unsigned long value)
{
 int exp, rnd;

 exp = rnd = 0;
 while (value > MAXFRACT) {
  rnd = value & (1 << (EXPSIZE - 1));
  value >>= EXPSIZE;
  exp++;
 }




 if (rnd && (++value > MAXFRACT)) {
  value >>= EXPSIZE;
  exp++;
 }




 exp <<= MANTSIZE;
 exp += value;
 return exp;
}


static comp2_t encode_comp2_t(u64 value)
{
 int exp, rnd;

 exp = (value > (MAXFRACT2>>1));
 rnd = 0;
 while (value > MAXFRACT2) {
  rnd = value & 1;
  value >>= 1;
  exp++;
 }




 if (rnd && (++value > MAXFRACT2)) {
  value >>= 1;
  exp++;
 }

 if (exp > MAXEXP2) {

  return (1ul << (MANTSIZE2+EXPSIZE2-1)) - 1;
 } else {
  return (value & (MAXFRACT2>>1)) | (exp << (MANTSIZE2-1));
 }
}




static u32 encode_float(u64 value)
{
 unsigned exp = 190;
 unsigned u;

 if (value == 0)
  return 0;
 while ((s64)value > 0) {
  value <<= 1;
  exp--;
 }
 u = (u32)(value >> 40) & 0x7fffffu;
 return u | (exp << 23);
}
static void fill_ac(acct_t *ac)
{
 struct pacct_struct *pacct = &current->signal->pacct;
 u64 elapsed, run_time;
 struct tty_struct *tty;





 memset(ac, 0, sizeof(acct_t));

 ac->ac_version = ACCT_VERSION | ACCT_BYTEORDER;
 strlcpy(ac->ac_comm, current->comm, sizeof(ac->ac_comm));


 run_time = ktime_get_ns();
 run_time -= current->group_leader->start_time;

 elapsed = nsec_to_AHZ(run_time);
 ac->ac_etime = encode_float(elapsed);
 ac->ac_etime = encode_comp_t(elapsed < (unsigned long) -1l ?
    (unsigned long) elapsed : (unsigned long) -1l);
 {

  comp2_t etime = encode_comp2_t(elapsed);

  ac->ac_etime_hi = etime >> 16;
  ac->ac_etime_lo = (u16) etime;
 }
 do_div(elapsed, AHZ);
 ac->ac_btime = get_seconds() - elapsed;
 ac->ac_ahz = AHZ;

 spin_lock_irq(&current->sighand->siglock);
 tty = current->signal->tty;
 ac->ac_tty = tty ? old_encode_dev(tty_devnum(tty)) : 0;
 ac->ac_utime = encode_comp_t(jiffies_to_AHZ(cputime_to_jiffies(pacct->ac_utime)));
 ac->ac_stime = encode_comp_t(jiffies_to_AHZ(cputime_to_jiffies(pacct->ac_stime)));
 ac->ac_flag = pacct->ac_flag;
 ac->ac_mem = encode_comp_t(pacct->ac_mem);
 ac->ac_minflt = encode_comp_t(pacct->ac_minflt);
 ac->ac_majflt = encode_comp_t(pacct->ac_majflt);
 ac->ac_exitcode = pacct->ac_exitcode;
 spin_unlock_irq(&current->sighand->siglock);
}



static void do_acct_process(struct bsd_acct_struct *acct)
{
 acct_t ac;
 unsigned long flim;
 const struct cred *orig_cred;
 struct file *file = acct->file;




 flim = current->signal->rlim[RLIMIT_FSIZE].rlim_cur;
 current->signal->rlim[RLIMIT_FSIZE].rlim_cur = RLIM_INFINITY;

 orig_cred = override_creds(file->f_cred);





 if (!check_free_space(acct))
  goto out;

 fill_ac(&ac);

 ac.ac_uid = from_kuid_munged(file->f_cred->user_ns, orig_cred->uid);
 ac.ac_gid = from_kgid_munged(file->f_cred->user_ns, orig_cred->gid);

 ac.ac_uid16 = ac.ac_uid;
 ac.ac_gid16 = ac.ac_gid;
 {
  struct pid_namespace *ns = acct->ns;

  ac.ac_pid = task_tgid_nr_ns(current, ns);
  rcu_read_lock();
  ac.ac_ppid = task_tgid_nr_ns(rcu_dereference(current->real_parent),
          ns);
  rcu_read_unlock();
 }




 if (file_start_write_trylock(file)) {

  loff_t pos = 0;
  __kernel_write(file, (char *)&ac, sizeof(acct_t), &pos);
  file_end_write(file);
 }
out:
 current->signal->rlim[RLIMIT_FSIZE].rlim_cur = flim;
 revert_creds(orig_cred);
}






void acct_collect(long exitcode, int group_dead)
{
 struct pacct_struct *pacct = &current->signal->pacct;
 cputime_t utime, stime;
 unsigned long vsize = 0;

 if (group_dead && current->mm) {
  struct vm_area_struct *vma;

  down_read(&current->mm->mmap_sem);
  vma = current->mm->mmap;
  while (vma) {
   vsize += vma->vm_end - vma->vm_start;
   vma = vma->vm_next;
  }
  up_read(&current->mm->mmap_sem);
 }

 spin_lock_irq(&current->sighand->siglock);
 if (group_dead)
  pacct->ac_mem = vsize / 1024;
 if (thread_group_leader(current)) {
  pacct->ac_exitcode = exitcode;
  if (current->flags & PF_FORKNOEXEC)
   pacct->ac_flag |= AFORK;
 }
 if (current->flags & PF_SUPERPRIV)
  pacct->ac_flag |= ASU;
 if (current->flags & PF_DUMPCORE)
  pacct->ac_flag |= ACORE;
 if (current->flags & PF_SIGNALED)
  pacct->ac_flag |= AXSIG;
 task_cputime(current, &utime, &stime);
 pacct->ac_utime += utime;
 pacct->ac_stime += stime;
 pacct->ac_minflt += current->min_flt;
 pacct->ac_majflt += current->maj_flt;
 spin_unlock_irq(&current->sighand->siglock);
}

static void slow_acct_process(struct pid_namespace *ns)
{
 for ( ; ns; ns = ns->parent) {
  struct bsd_acct_struct *acct = acct_get(ns);
  if (acct) {
   do_acct_process(acct);
   mutex_unlock(&acct->lock);
   acct_put(acct);
  }
 }
}






void acct_process(void)
{
 struct pid_namespace *ns;






 for (ns = task_active_pid_ns(current); ns != NULL; ns = ns->parent) {
  if (ns->bacct)
   break;
 }
 if (unlikely(ns))
  slow_acct_process(ns);
}
static struct alarm_base {
 spinlock_t lock;
 struct timerqueue_head timerqueue;
 ktime_t (*gettime)(void);
 clockid_t base_clockid;
} alarm_bases[ALARM_NUMTYPE];


static ktime_t freezer_delta;
static DEFINE_SPINLOCK(freezer_delta_lock);

static struct wakeup_source *ws;


static struct rtc_timer rtctimer;
static struct rtc_device *rtcdev;
static DEFINE_SPINLOCK(rtcdev_lock);
struct rtc_device *alarmtimer_get_rtcdev(void)
{
 unsigned long flags;
 struct rtc_device *ret;

 spin_lock_irqsave(&rtcdev_lock, flags);
 ret = rtcdev;
 spin_unlock_irqrestore(&rtcdev_lock, flags);

 return ret;
}
EXPORT_SYMBOL_GPL(alarmtimer_get_rtcdev);

static int alarmtimer_rtc_add_device(struct device *dev,
    struct class_interface *class_intf)
{
 unsigned long flags;
 struct rtc_device *rtc = to_rtc_device(dev);

 if (rtcdev)
  return -EBUSY;

 if (!rtc->ops->set_alarm)
  return -1;
 if (!device_may_wakeup(rtc->dev.parent))
  return -1;

 spin_lock_irqsave(&rtcdev_lock, flags);
 if (!rtcdev) {
  rtcdev = rtc;

  get_device(dev);
 }
 spin_unlock_irqrestore(&rtcdev_lock, flags);
 return 0;
}

static inline void alarmtimer_rtc_timer_init(void)
{
 rtc_timer_init(&rtctimer, NULL, NULL);
}

static struct class_interface alarmtimer_rtc_interface = {
 .add_dev = &alarmtimer_rtc_add_device,
};

static int alarmtimer_rtc_interface_setup(void)
{
 alarmtimer_rtc_interface.class = rtc_class;
 return class_interface_register(&alarmtimer_rtc_interface);
}
static void alarmtimer_rtc_interface_remove(void)
{
 class_interface_unregister(&alarmtimer_rtc_interface);
}
struct rtc_device *alarmtimer_get_rtcdev(void)
{
 return NULL;
}
static inline int alarmtimer_rtc_interface_setup(void) { return 0; }
static inline void alarmtimer_rtc_interface_remove(void) { }
static inline void alarmtimer_rtc_timer_init(void) { }
static void alarmtimer_enqueue(struct alarm_base *base, struct alarm *alarm)
{
 if (alarm->state & ALARMTIMER_STATE_ENQUEUED)
  timerqueue_del(&base->timerqueue, &alarm->node);

 timerqueue_add(&base->timerqueue, &alarm->node);
 alarm->state |= ALARMTIMER_STATE_ENQUEUED;
}
static void alarmtimer_dequeue(struct alarm_base *base, struct alarm *alarm)
{
 if (!(alarm->state & ALARMTIMER_STATE_ENQUEUED))
  return;

 timerqueue_del(&base->timerqueue, &alarm->node);
 alarm->state &= ~ALARMTIMER_STATE_ENQUEUED;
}
static enum hrtimer_restart alarmtimer_fired(struct hrtimer *timer)
{
 struct alarm *alarm = container_of(timer, struct alarm, timer);
 struct alarm_base *base = &alarm_bases[alarm->type];
 unsigned long flags;
 int ret = HRTIMER_NORESTART;
 int restart = ALARMTIMER_NORESTART;

 spin_lock_irqsave(&base->lock, flags);
 alarmtimer_dequeue(base, alarm);
 spin_unlock_irqrestore(&base->lock, flags);

 if (alarm->function)
  restart = alarm->function(alarm, base->gettime());

 spin_lock_irqsave(&base->lock, flags);
 if (restart != ALARMTIMER_NORESTART) {
  hrtimer_set_expires(&alarm->timer, alarm->node.expires);
  alarmtimer_enqueue(base, alarm);
  ret = HRTIMER_RESTART;
 }
 spin_unlock_irqrestore(&base->lock, flags);

 return ret;

}

ktime_t alarm_expires_remaining(const struct alarm *alarm)
{
 struct alarm_base *base = &alarm_bases[alarm->type];
 return ktime_sub(alarm->node.expires, base->gettime());
}
EXPORT_SYMBOL_GPL(alarm_expires_remaining);

static int alarmtimer_suspend(struct device *dev)
{
 struct rtc_time tm;
 ktime_t min, now;
 unsigned long flags;
 struct rtc_device *rtc;
 int i;
 int ret;

 spin_lock_irqsave(&freezer_delta_lock, flags);
 min = freezer_delta;
 freezer_delta = ktime_set(0, 0);
 spin_unlock_irqrestore(&freezer_delta_lock, flags);

 rtc = alarmtimer_get_rtcdev();

 if (!rtc)
  return 0;


 for (i = 0; i < ALARM_NUMTYPE; i++) {
  struct alarm_base *base = &alarm_bases[i];
  struct timerqueue_node *next;
  ktime_t delta;

  spin_lock_irqsave(&base->lock, flags);
  next = timerqueue_getnext(&base->timerqueue);
  spin_unlock_irqrestore(&base->lock, flags);
  if (!next)
   continue;
  delta = ktime_sub(next->expires, base->gettime());
  if (!min.tv64 || (delta.tv64 < min.tv64))
   min = delta;
 }
 if (min.tv64 == 0)
  return 0;

 if (ktime_to_ns(min) < 2 * NSEC_PER_SEC) {
  __pm_wakeup_event(ws, 2 * MSEC_PER_SEC);
  return -EBUSY;
 }


 rtc_timer_cancel(rtc, &rtctimer);
 rtc_read_time(rtc, &tm);
 now = rtc_tm_to_ktime(tm);
 now = ktime_add(now, min);


 ret = rtc_timer_start(rtc, &rtctimer, now, ktime_set(0, 0));
 if (ret < 0)
  __pm_wakeup_event(ws, MSEC_PER_SEC);
 return ret;
}

static int alarmtimer_resume(struct device *dev)
{
 struct rtc_device *rtc;

 rtc = alarmtimer_get_rtcdev();
 if (rtc)
  rtc_timer_cancel(rtc, &rtctimer);
 return 0;
}

static int alarmtimer_suspend(struct device *dev)
{
 return 0;
}

static int alarmtimer_resume(struct device *dev)
{
 return 0;
}

static void alarmtimer_freezerset(ktime_t absexp, enum alarmtimer_type type)
{
 ktime_t delta;
 unsigned long flags;
 struct alarm_base *base = &alarm_bases[type];

 delta = ktime_sub(absexp, base->gettime());

 spin_lock_irqsave(&freezer_delta_lock, flags);
 if (!freezer_delta.tv64 || (delta.tv64 < freezer_delta.tv64))
  freezer_delta = delta;
 spin_unlock_irqrestore(&freezer_delta_lock, flags);
}
void alarm_init(struct alarm *alarm, enum alarmtimer_type type,
  enum alarmtimer_restart (*function)(struct alarm *, ktime_t))
{
 timerqueue_init(&alarm->node);
 hrtimer_init(&alarm->timer, alarm_bases[type].base_clockid,
   HRTIMER_MODE_ABS);
 alarm->timer.function = alarmtimer_fired;
 alarm->function = function;
 alarm->type = type;
 alarm->state = ALARMTIMER_STATE_INACTIVE;
}
EXPORT_SYMBOL_GPL(alarm_init);






void alarm_start(struct alarm *alarm, ktime_t start)
{
 struct alarm_base *base = &alarm_bases[alarm->type];
 unsigned long flags;

 spin_lock_irqsave(&base->lock, flags);
 alarm->node.expires = start;
 alarmtimer_enqueue(base, alarm);
 hrtimer_start(&alarm->timer, alarm->node.expires, HRTIMER_MODE_ABS);
 spin_unlock_irqrestore(&base->lock, flags);
}
EXPORT_SYMBOL_GPL(alarm_start);






void alarm_start_relative(struct alarm *alarm, ktime_t start)
{
 struct alarm_base *base = &alarm_bases[alarm->type];

 start = ktime_add(start, base->gettime());
 alarm_start(alarm, start);
}
EXPORT_SYMBOL_GPL(alarm_start_relative);

void alarm_restart(struct alarm *alarm)
{
 struct alarm_base *base = &alarm_bases[alarm->type];
 unsigned long flags;

 spin_lock_irqsave(&base->lock, flags);
 hrtimer_set_expires(&alarm->timer, alarm->node.expires);
 hrtimer_restart(&alarm->timer);
 alarmtimer_enqueue(base, alarm);
 spin_unlock_irqrestore(&base->lock, flags);
}
EXPORT_SYMBOL_GPL(alarm_restart);
int alarm_try_to_cancel(struct alarm *alarm)
{
 struct alarm_base *base = &alarm_bases[alarm->type];
 unsigned long flags;
 int ret;

 spin_lock_irqsave(&base->lock, flags);
 ret = hrtimer_try_to_cancel(&alarm->timer);
 if (ret >= 0)
  alarmtimer_dequeue(base, alarm);
 spin_unlock_irqrestore(&base->lock, flags);
 return ret;
}
EXPORT_SYMBOL_GPL(alarm_try_to_cancel);
int alarm_cancel(struct alarm *alarm)
{
 for (;;) {
  int ret = alarm_try_to_cancel(alarm);
  if (ret >= 0)
   return ret;
  cpu_relax();
 }
}
EXPORT_SYMBOL_GPL(alarm_cancel);


u64 alarm_forward(struct alarm *alarm, ktime_t now, ktime_t interval)
{
 u64 overrun = 1;
 ktime_t delta;

 delta = ktime_sub(now, alarm->node.expires);

 if (delta.tv64 < 0)
  return 0;

 if (unlikely(delta.tv64 >= interval.tv64)) {
  s64 incr = ktime_to_ns(interval);

  overrun = ktime_divns(delta, incr);

  alarm->node.expires = ktime_add_ns(alarm->node.expires,
       incr*overrun);

  if (alarm->node.expires.tv64 > now.tv64)
   return overrun;




  overrun++;
 }

 alarm->node.expires = ktime_add(alarm->node.expires, interval);
 return overrun;
}
EXPORT_SYMBOL_GPL(alarm_forward);

u64 alarm_forward_now(struct alarm *alarm, ktime_t interval)
{
 struct alarm_base *base = &alarm_bases[alarm->type];

 return alarm_forward(alarm, base->gettime(), interval);
}
EXPORT_SYMBOL_GPL(alarm_forward_now);






static enum alarmtimer_type clock2alarm(clockid_t clockid)
{
 if (clockid == CLOCK_REALTIME_ALARM)
  return ALARM_REALTIME;
 if (clockid == CLOCK_BOOTTIME_ALARM)
  return ALARM_BOOTTIME;
 return -1;
}







static enum alarmtimer_restart alarm_handle_timer(struct alarm *alarm,
       ktime_t now)
{
 unsigned long flags;
 struct k_itimer *ptr = container_of(alarm, struct k_itimer,
      it.alarm.alarmtimer);
 enum alarmtimer_restart result = ALARMTIMER_NORESTART;

 spin_lock_irqsave(&ptr->it_lock, flags);
 if ((ptr->it_sigev_notify & ~SIGEV_THREAD_ID) != SIGEV_NONE) {
  if (posix_timer_event(ptr, 0) != 0)
   ptr->it_overrun++;
 }


 if (ptr->it.alarm.interval.tv64) {
  ptr->it_overrun += alarm_forward(alarm, now,
      ptr->it.alarm.interval);
  result = ALARMTIMER_RESTART;
 }
 spin_unlock_irqrestore(&ptr->it_lock, flags);

 return result;
}
static int alarm_clock_getres(const clockid_t which_clock, struct timespec *tp)
{
 if (!alarmtimer_get_rtcdev())
  return -EINVAL;

 tp->tv_sec = 0;
 tp->tv_nsec = hrtimer_resolution;
 return 0;
}
static int alarm_clock_get(clockid_t which_clock, struct timespec *tp)
{
 struct alarm_base *base = &alarm_bases[clock2alarm(which_clock)];

 if (!alarmtimer_get_rtcdev())
  return -EINVAL;

 *tp = ktime_to_timespec(base->gettime());
 return 0;
}







static int alarm_timer_create(struct k_itimer *new_timer)
{
 enum alarmtimer_type type;
 struct alarm_base *base;

 if (!alarmtimer_get_rtcdev())
  return -ENOTSUPP;

 if (!capable(CAP_WAKE_ALARM))
  return -EPERM;

 type = clock2alarm(new_timer->it_clock);
 base = &alarm_bases[type];
 alarm_init(&new_timer->it.alarm.alarmtimer, type, alarm_handle_timer);
 return 0;
}
static void alarm_timer_get(struct k_itimer *timr,
    struct itimerspec *cur_setting)
{
 ktime_t relative_expiry_time =
  alarm_expires_remaining(&(timr->it.alarm.alarmtimer));

 if (ktime_to_ns(relative_expiry_time) > 0) {
  cur_setting->it_value = ktime_to_timespec(relative_expiry_time);
 } else {
  cur_setting->it_value.tv_sec = 0;
  cur_setting->it_value.tv_nsec = 0;
 }

 cur_setting->it_interval = ktime_to_timespec(timr->it.alarm.interval);
}







static int alarm_timer_del(struct k_itimer *timr)
{
 if (!rtcdev)
  return -ENOTSUPP;

 if (alarm_try_to_cancel(&timr->it.alarm.alarmtimer) < 0)
  return TIMER_RETRY;

 return 0;
}
static int alarm_timer_set(struct k_itimer *timr, int flags,
    struct itimerspec *new_setting,
    struct itimerspec *old_setting)
{
 ktime_t exp;

 if (!rtcdev)
  return -ENOTSUPP;

 if (flags & ~TIMER_ABSTIME)
  return -EINVAL;

 if (old_setting)
  alarm_timer_get(timr, old_setting);


 if (alarm_try_to_cancel(&timr->it.alarm.alarmtimer) < 0)
  return TIMER_RETRY;


 timr->it.alarm.interval = timespec_to_ktime(new_setting->it_interval);
 exp = timespec_to_ktime(new_setting->it_value);

 if (flags != TIMER_ABSTIME) {
  ktime_t now;

  now = alarm_bases[timr->it.alarm.alarmtimer.type].gettime();
  exp = ktime_add(now, exp);
 }

 alarm_start(&timr->it.alarm.alarmtimer, exp);
 return 0;
}







static enum alarmtimer_restart alarmtimer_nsleep_wakeup(struct alarm *alarm,
        ktime_t now)
{
 struct task_struct *task = (struct task_struct *)alarm->data;

 alarm->data = NULL;
 if (task)
  wake_up_process(task);
 return ALARMTIMER_NORESTART;
}
static int alarmtimer_do_nsleep(struct alarm *alarm, ktime_t absexp)
{
 alarm->data = (void *)current;
 do {
  set_current_state(TASK_INTERRUPTIBLE);
  alarm_start(alarm, absexp);
  if (likely(alarm->data))
   schedule();

  alarm_cancel(alarm);
 } while (alarm->data && !signal_pending(current));

 __set_current_state(TASK_RUNNING);

 return (alarm->data == NULL);
}
static int update_rmtp(ktime_t exp, enum alarmtimer_type type,
   struct timespec __user *rmtp)
{
 struct timespec rmt;
 ktime_t rem;

 rem = ktime_sub(exp, alarm_bases[type].gettime());

 if (rem.tv64 <= 0)
  return 0;
 rmt = ktime_to_timespec(rem);

 if (copy_to_user(rmtp, &rmt, sizeof(*rmtp)))
  return -EFAULT;

 return 1;

}







static long __sched alarm_timer_nsleep_restart(struct restart_block *restart)
{
 enum alarmtimer_type type = restart->nanosleep.clockid;
 ktime_t exp;
 struct timespec __user *rmtp;
 struct alarm alarm;
 int ret = 0;

 exp.tv64 = restart->nanosleep.expires;
 alarm_init(&alarm, type, alarmtimer_nsleep_wakeup);

 if (alarmtimer_do_nsleep(&alarm, exp))
  goto out;

 if (freezing(current))
  alarmtimer_freezerset(exp, type);

 rmtp = restart->nanosleep.rmtp;
 if (rmtp) {
  ret = update_rmtp(exp, type, rmtp);
  if (ret <= 0)
   goto out;
 }



 ret = -ERESTART_RESTARTBLOCK;
out:
 return ret;
}
static int alarm_timer_nsleep(const clockid_t which_clock, int flags,
       struct timespec *tsreq, struct timespec __user *rmtp)
{
 enum alarmtimer_type type = clock2alarm(which_clock);
 struct alarm alarm;
 ktime_t exp;
 int ret = 0;
 struct restart_block *restart;

 if (!alarmtimer_get_rtcdev())
  return -ENOTSUPP;

 if (flags & ~TIMER_ABSTIME)
  return -EINVAL;

 if (!capable(CAP_WAKE_ALARM))
  return -EPERM;

 alarm_init(&alarm, type, alarmtimer_nsleep_wakeup);

 exp = timespec_to_ktime(*tsreq);

 if (flags != TIMER_ABSTIME) {
  ktime_t now = alarm_bases[type].gettime();
  exp = ktime_add(now, exp);
 }

 if (alarmtimer_do_nsleep(&alarm, exp))
  goto out;

 if (freezing(current))
  alarmtimer_freezerset(exp, type);


 if (flags == TIMER_ABSTIME) {
  ret = -ERESTARTNOHAND;
  goto out;
 }

 if (rmtp) {
  ret = update_rmtp(exp, type, rmtp);
  if (ret <= 0)
   goto out;
 }

 restart = &current->restart_block;
 restart->fn = alarm_timer_nsleep_restart;
 restart->nanosleep.clockid = type;
 restart->nanosleep.expires = exp.tv64;
 restart->nanosleep.rmtp = rmtp;
 ret = -ERESTART_RESTARTBLOCK;

out:
 return ret;
}



static const struct dev_pm_ops alarmtimer_pm_ops = {
 .suspend = alarmtimer_suspend,
 .resume = alarmtimer_resume,
};

static struct platform_driver alarmtimer_driver = {
 .driver = {
  .name = "alarmtimer",
  .pm = &alarmtimer_pm_ops,
 }
};







static int __init alarmtimer_init(void)
{
 struct platform_device *pdev;
 int error = 0;
 int i;
 struct k_clock alarm_clock = {
  .clock_getres = alarm_clock_getres,
  .clock_get = alarm_clock_get,
  .timer_create = alarm_timer_create,
  .timer_set = alarm_timer_set,
  .timer_del = alarm_timer_del,
  .timer_get = alarm_timer_get,
  .nsleep = alarm_timer_nsleep,
 };

 alarmtimer_rtc_timer_init();

 posix_timers_register_clock(CLOCK_REALTIME_ALARM, &alarm_clock);
 posix_timers_register_clock(CLOCK_BOOTTIME_ALARM, &alarm_clock);


 alarm_bases[ALARM_REALTIME].base_clockid = CLOCK_REALTIME;
 alarm_bases[ALARM_REALTIME].gettime = &ktime_get_real;
 alarm_bases[ALARM_BOOTTIME].base_clockid = CLOCK_BOOTTIME;
 alarm_bases[ALARM_BOOTTIME].gettime = &ktime_get_boottime;
 for (i = 0; i < ALARM_NUMTYPE; i++) {
  timerqueue_init_head(&alarm_bases[i].timerqueue);
  spin_lock_init(&alarm_bases[i].lock);
 }

 error = alarmtimer_rtc_interface_setup();
 if (error)
  return error;

 error = platform_driver_register(&alarmtimer_driver);
 if (error)
  goto out_if;

 pdev = platform_device_register_simple("alarmtimer", -1, NULL, 0);
 if (IS_ERR(pdev)) {
  error = PTR_ERR(pdev);
  goto out_drv;
 }
 ws = wakeup_source_register("alarmtimer");
 return 0;

out_drv:
 platform_driver_unregister(&alarmtimer_driver);
out_if:
 alarmtimer_rtc_interface_remove();
 return error;
}
device_initcall(alarmtimer_init);

static void bpf_array_free_percpu(struct bpf_array *array)
{
 int i;

 for (i = 0; i < array->map.max_entries; i++)
  free_percpu(array->pptrs[i]);
}

static int bpf_array_alloc_percpu(struct bpf_array *array)
{
 void __percpu *ptr;
 int i;

 for (i = 0; i < array->map.max_entries; i++) {
  ptr = __alloc_percpu_gfp(array->elem_size, 8,
      GFP_USER | __GFP_NOWARN);
  if (!ptr) {
   bpf_array_free_percpu(array);
   return -ENOMEM;
  }
  array->pptrs[i] = ptr;
 }

 return 0;
}


static struct bpf_map *array_map_alloc(union bpf_attr *attr)
{
 bool percpu = attr->map_type == BPF_MAP_TYPE_PERCPU_ARRAY;
 struct bpf_array *array;
 u64 array_size;
 u32 elem_size;


 if (attr->max_entries == 0 || attr->key_size != 4 ||
     attr->value_size == 0 || attr->map_flags)
  return ERR_PTR(-EINVAL);

 if (attr->value_size >= 1 << (KMALLOC_SHIFT_MAX - 1))



  return ERR_PTR(-E2BIG);

 elem_size = round_up(attr->value_size, 8);

 array_size = sizeof(*array);
 if (percpu)
  array_size += (u64) attr->max_entries * sizeof(void *);
 else
  array_size += (u64) attr->max_entries * elem_size;


 if (array_size >= U32_MAX - PAGE_SIZE)
  return ERR_PTR(-ENOMEM);



 array = kzalloc(array_size, GFP_USER | __GFP_NOWARN);
 if (!array) {
  array = vzalloc(array_size);
  if (!array)
   return ERR_PTR(-ENOMEM);
 }


 array->map.map_type = attr->map_type;
 array->map.key_size = attr->key_size;
 array->map.value_size = attr->value_size;
 array->map.max_entries = attr->max_entries;
 array->elem_size = elem_size;

 if (!percpu)
  goto out;

 array_size += (u64) attr->max_entries * elem_size * num_possible_cpus();

 if (array_size >= U32_MAX - PAGE_SIZE ||
     elem_size > PCPU_MIN_UNIT_SIZE || bpf_array_alloc_percpu(array)) {
  kvfree(array);
  return ERR_PTR(-ENOMEM);
 }
out:
 array->map.pages = round_up(array_size, PAGE_SIZE) >> PAGE_SHIFT;

 return &array->map;
}


static void *array_map_lookup_elem(struct bpf_map *map, void *key)
{
 struct bpf_array *array = container_of(map, struct bpf_array, map);
 u32 index = *(u32 *)key;

 if (unlikely(index >= array->map.max_entries))
  return NULL;

 return array->value + array->elem_size * index;
}


static void *percpu_array_map_lookup_elem(struct bpf_map *map, void *key)
{
 struct bpf_array *array = container_of(map, struct bpf_array, map);
 u32 index = *(u32 *)key;

 if (unlikely(index >= array->map.max_entries))
  return NULL;

 return this_cpu_ptr(array->pptrs[index]);
}

int bpf_percpu_array_copy(struct bpf_map *map, void *key, void *value)
{
 struct bpf_array *array = container_of(map, struct bpf_array, map);
 u32 index = *(u32 *)key;
 void __percpu *pptr;
 int cpu, off = 0;
 u32 size;

 if (unlikely(index >= array->map.max_entries))
  return -ENOENT;





 size = round_up(map->value_size, 8);
 rcu_read_lock();
 pptr = array->pptrs[index];
 for_each_possible_cpu(cpu) {
  bpf_long_memcpy(value + off, per_cpu_ptr(pptr, cpu), size);
  off += size;
 }
 rcu_read_unlock();
 return 0;
}


static int array_map_get_next_key(struct bpf_map *map, void *key, void *next_key)
{
 struct bpf_array *array = container_of(map, struct bpf_array, map);
 u32 index = *(u32 *)key;
 u32 *next = (u32 *)next_key;

 if (index >= array->map.max_entries) {
  *next = 0;
  return 0;
 }

 if (index == array->map.max_entries - 1)
  return -ENOENT;

 *next = index + 1;
 return 0;
}


static int array_map_update_elem(struct bpf_map *map, void *key, void *value,
     u64 map_flags)
{
 struct bpf_array *array = container_of(map, struct bpf_array, map);
 u32 index = *(u32 *)key;

 if (unlikely(map_flags > BPF_EXIST))

  return -EINVAL;

 if (unlikely(index >= array->map.max_entries))

  return -E2BIG;

 if (unlikely(map_flags == BPF_NOEXIST))

  return -EEXIST;

 if (array->map.map_type == BPF_MAP_TYPE_PERCPU_ARRAY)
  memcpy(this_cpu_ptr(array->pptrs[index]),
         value, map->value_size);
 else
  memcpy(array->value + array->elem_size * index,
         value, map->value_size);
 return 0;
}

int bpf_percpu_array_update(struct bpf_map *map, void *key, void *value,
       u64 map_flags)
{
 struct bpf_array *array = container_of(map, struct bpf_array, map);
 u32 index = *(u32 *)key;
 void __percpu *pptr;
 int cpu, off = 0;
 u32 size;

 if (unlikely(map_flags > BPF_EXIST))

  return -EINVAL;

 if (unlikely(index >= array->map.max_entries))

  return -E2BIG;

 if (unlikely(map_flags == BPF_NOEXIST))

  return -EEXIST;







 size = round_up(map->value_size, 8);
 rcu_read_lock();
 pptr = array->pptrs[index];
 for_each_possible_cpu(cpu) {
  bpf_long_memcpy(per_cpu_ptr(pptr, cpu), value + off, size);
  off += size;
 }
 rcu_read_unlock();
 return 0;
}


static int array_map_delete_elem(struct bpf_map *map, void *key)
{
 return -EINVAL;
}


static void array_map_free(struct bpf_map *map)
{
 struct bpf_array *array = container_of(map, struct bpf_array, map);






 synchronize_rcu();

 if (array->map.map_type == BPF_MAP_TYPE_PERCPU_ARRAY)
  bpf_array_free_percpu(array);

 kvfree(array);
}

static const struct bpf_map_ops array_ops = {
 .map_alloc = array_map_alloc,
 .map_free = array_map_free,
 .map_get_next_key = array_map_get_next_key,
 .map_lookup_elem = array_map_lookup_elem,
 .map_update_elem = array_map_update_elem,
 .map_delete_elem = array_map_delete_elem,
};

static struct bpf_map_type_list array_type __read_mostly = {
 .ops = &array_ops,
 .type = BPF_MAP_TYPE_ARRAY,
};

static const struct bpf_map_ops percpu_array_ops = {
 .map_alloc = array_map_alloc,
 .map_free = array_map_free,
 .map_get_next_key = array_map_get_next_key,
 .map_lookup_elem = percpu_array_map_lookup_elem,
 .map_update_elem = array_map_update_elem,
 .map_delete_elem = array_map_delete_elem,
};

static struct bpf_map_type_list percpu_array_type __read_mostly = {
 .ops = &percpu_array_ops,
 .type = BPF_MAP_TYPE_PERCPU_ARRAY,
};

static int __init register_array_map(void)
{
 bpf_register_map_type(&array_type);
 bpf_register_map_type(&percpu_array_type);
 return 0;
}
late_initcall(register_array_map);

static struct bpf_map *fd_array_map_alloc(union bpf_attr *attr)
{

 if (attr->value_size != sizeof(u32))
  return ERR_PTR(-EINVAL);
 return array_map_alloc(attr);
}

static void fd_array_map_free(struct bpf_map *map)
{
 struct bpf_array *array = container_of(map, struct bpf_array, map);
 int i;

 synchronize_rcu();


 for (i = 0; i < array->map.max_entries; i++)
  BUG_ON(array->ptrs[i] != NULL);
 kvfree(array);
}

static void *fd_array_map_lookup_elem(struct bpf_map *map, void *key)
{
 return NULL;
}


static int fd_array_map_update_elem(struct bpf_map *map, void *key,
        void *value, u64 map_flags)
{
 struct bpf_array *array = container_of(map, struct bpf_array, map);
 void *new_ptr, *old_ptr;
 u32 index = *(u32 *)key, ufd;

 if (map_flags != BPF_ANY)
  return -EINVAL;

 if (index >= array->map.max_entries)
  return -E2BIG;

 ufd = *(u32 *)value;
 new_ptr = map->ops->map_fd_get_ptr(map, ufd);
 if (IS_ERR(new_ptr))
  return PTR_ERR(new_ptr);

 old_ptr = xchg(array->ptrs + index, new_ptr);
 if (old_ptr)
  map->ops->map_fd_put_ptr(old_ptr);

 return 0;
}

static int fd_array_map_delete_elem(struct bpf_map *map, void *key)
{
 struct bpf_array *array = container_of(map, struct bpf_array, map);
 void *old_ptr;
 u32 index = *(u32 *)key;

 if (index >= array->map.max_entries)
  return -E2BIG;

 old_ptr = xchg(array->ptrs + index, NULL);
 if (old_ptr) {
  map->ops->map_fd_put_ptr(old_ptr);
  return 0;
 } else {
  return -ENOENT;
 }
}

static void *prog_fd_array_get_ptr(struct bpf_map *map, int fd)
{
 struct bpf_array *array = container_of(map, struct bpf_array, map);
 struct bpf_prog *prog = bpf_prog_get(fd);
 if (IS_ERR(prog))
  return prog;

 if (!bpf_prog_array_compatible(array, prog)) {
  bpf_prog_put(prog);
  return ERR_PTR(-EINVAL);
 }
 return prog;
}

static void prog_fd_array_put_ptr(void *ptr)
{
 struct bpf_prog *prog = ptr;

 bpf_prog_put_rcu(prog);
}


void bpf_fd_array_map_clear(struct bpf_map *map)
{
 struct bpf_array *array = container_of(map, struct bpf_array, map);
 int i;

 for (i = 0; i < array->map.max_entries; i++)
  fd_array_map_delete_elem(map, &i);
}

static const struct bpf_map_ops prog_array_ops = {
 .map_alloc = fd_array_map_alloc,
 .map_free = fd_array_map_free,
 .map_get_next_key = array_map_get_next_key,
 .map_lookup_elem = fd_array_map_lookup_elem,
 .map_update_elem = fd_array_map_update_elem,
 .map_delete_elem = fd_array_map_delete_elem,
 .map_fd_get_ptr = prog_fd_array_get_ptr,
 .map_fd_put_ptr = prog_fd_array_put_ptr,
};

static struct bpf_map_type_list prog_array_type __read_mostly = {
 .ops = &prog_array_ops,
 .type = BPF_MAP_TYPE_PROG_ARRAY,
};

static int __init register_prog_array_map(void)
{
 bpf_register_map_type(&prog_array_type);
 return 0;
}
late_initcall(register_prog_array_map);

static void perf_event_array_map_free(struct bpf_map *map)
{
 bpf_fd_array_map_clear(map);
 fd_array_map_free(map);
}

static void *perf_event_fd_array_get_ptr(struct bpf_map *map, int fd)
{
 struct perf_event *event;
 const struct perf_event_attr *attr;
 struct file *file;

 file = perf_event_get(fd);
 if (IS_ERR(file))
  return file;

 event = file->private_data;

 attr = perf_event_attrs(event);
 if (IS_ERR(attr))
  goto err;

 if (attr->inherit)
  goto err;

 if (attr->type == PERF_TYPE_RAW)
  return file;

 if (attr->type == PERF_TYPE_HARDWARE)
  return file;

 if (attr->type == PERF_TYPE_SOFTWARE &&
     attr->config == PERF_COUNT_SW_BPF_OUTPUT)
  return file;
err:
 fput(file);
 return ERR_PTR(-EINVAL);
}

static void perf_event_fd_array_put_ptr(void *ptr)
{
 fput((struct file *)ptr);
}

static const struct bpf_map_ops perf_event_array_ops = {
 .map_alloc = fd_array_map_alloc,
 .map_free = perf_event_array_map_free,
 .map_get_next_key = array_map_get_next_key,
 .map_lookup_elem = fd_array_map_lookup_elem,
 .map_update_elem = fd_array_map_update_elem,
 .map_delete_elem = fd_array_map_delete_elem,
 .map_fd_get_ptr = perf_event_fd_array_get_ptr,
 .map_fd_put_ptr = perf_event_fd_array_put_ptr,
};

static struct bpf_map_type_list perf_event_array_type __read_mostly = {
 .ops = &perf_event_array_ops,
 .type = BPF_MAP_TYPE_PERF_EVENT_ARRAY,
};

static int __init register_perf_event_array_map(void)
{
 bpf_register_map_type(&perf_event_array_type);
 return 0;
}
late_initcall(register_perf_event_array_map);


static async_cookie_t next_cookie = 1;


static LIST_HEAD(async_global_pending);
static ASYNC_DOMAIN(async_dfl_domain);
static DEFINE_SPINLOCK(async_lock);

struct async_entry {
 struct list_head domain_list;
 struct list_head global_list;
 struct work_struct work;
 async_cookie_t cookie;
 async_func_t func;
 void *data;
 struct async_domain *domain;
};

static DECLARE_WAIT_QUEUE_HEAD(async_done);

static atomic_t entry_count;

static async_cookie_t lowest_in_progress(struct async_domain *domain)
{
 struct list_head *pending;
 async_cookie_t ret = ASYNC_COOKIE_MAX;
 unsigned long flags;

 spin_lock_irqsave(&async_lock, flags);

 if (domain)
  pending = &domain->pending;
 else
  pending = &async_global_pending;

 if (!list_empty(pending))
  ret = list_first_entry(pending, struct async_entry,
           domain_list)->cookie;

 spin_unlock_irqrestore(&async_lock, flags);
 return ret;
}




static void async_run_entry_fn(struct work_struct *work)
{
 struct async_entry *entry =
  container_of(work, struct async_entry, work);
 unsigned long flags;
 ktime_t uninitialized_var(calltime), delta, rettime;


 if (initcall_debug && system_state == SYSTEM_BOOTING) {
  pr_debug("calling  %lli_%pF @ %i\n",
   (long long)entry->cookie,
   entry->func, task_pid_nr(current));
  calltime = ktime_get();
 }
 entry->func(entry->data, entry->cookie);
 if (initcall_debug && system_state == SYSTEM_BOOTING) {
  rettime = ktime_get();
  delta = ktime_sub(rettime, calltime);
  pr_debug("initcall %lli_%pF returned 0 after %lld usecs\n",
   (long long)entry->cookie,
   entry->func,
   (long long)ktime_to_ns(delta) >> 10);
 }


 spin_lock_irqsave(&async_lock, flags);
 list_del_init(&entry->domain_list);
 list_del_init(&entry->global_list);


 kfree(entry);
 atomic_dec(&entry_count);

 spin_unlock_irqrestore(&async_lock, flags);


 wake_up(&async_done);
}

static async_cookie_t __async_schedule(async_func_t func, void *data, struct async_domain *domain)
{
 struct async_entry *entry;
 unsigned long flags;
 async_cookie_t newcookie;


 entry = kzalloc(sizeof(struct async_entry), GFP_ATOMIC);





 if (!entry || atomic_read(&entry_count) > MAX_WORK) {
  kfree(entry);
  spin_lock_irqsave(&async_lock, flags);
  newcookie = next_cookie++;
  spin_unlock_irqrestore(&async_lock, flags);


  func(data, newcookie);
  return newcookie;
 }
 INIT_LIST_HEAD(&entry->domain_list);
 INIT_LIST_HEAD(&entry->global_list);
 INIT_WORK(&entry->work, async_run_entry_fn);
 entry->func = func;
 entry->data = data;
 entry->domain = domain;

 spin_lock_irqsave(&async_lock, flags);


 newcookie = entry->cookie = next_cookie++;

 list_add_tail(&entry->domain_list, &domain->pending);
 if (domain->registered)
  list_add_tail(&entry->global_list, &async_global_pending);

 atomic_inc(&entry_count);
 spin_unlock_irqrestore(&async_lock, flags);


 current->flags |= PF_USED_ASYNC;


 queue_work(system_unbound_wq, &entry->work);

 return newcookie;
}
async_cookie_t async_schedule(async_func_t func, void *data)
{
 return __async_schedule(func, data, &async_dfl_domain);
}
EXPORT_SYMBOL_GPL(async_schedule);
async_cookie_t async_schedule_domain(async_func_t func, void *data,
         struct async_domain *domain)
{
 return __async_schedule(func, data, domain);
}
EXPORT_SYMBOL_GPL(async_schedule_domain);






void async_synchronize_full(void)
{
 async_synchronize_full_domain(NULL);
}
EXPORT_SYMBOL_GPL(async_synchronize_full);
void async_unregister_domain(struct async_domain *domain)
{
 spin_lock_irq(&async_lock);
 WARN_ON(!domain->registered || !list_empty(&domain->pending));
 domain->registered = 0;
 spin_unlock_irq(&async_lock);
}
EXPORT_SYMBOL_GPL(async_unregister_domain);
void async_synchronize_full_domain(struct async_domain *domain)
{
 async_synchronize_cookie_domain(ASYNC_COOKIE_MAX, domain);
}
EXPORT_SYMBOL_GPL(async_synchronize_full_domain);
void async_synchronize_cookie_domain(async_cookie_t cookie, struct async_domain *domain)
{
 ktime_t uninitialized_var(starttime), delta, endtime;

 if (initcall_debug && system_state == SYSTEM_BOOTING) {
  pr_debug("async_waiting @ %i\n", task_pid_nr(current));
  starttime = ktime_get();
 }

 wait_event(async_done, lowest_in_progress(domain) >= cookie);

 if (initcall_debug && system_state == SYSTEM_BOOTING) {
  endtime = ktime_get();
  delta = ktime_sub(endtime, starttime);

  pr_debug("async_continuing @ %i after %lli usec\n",
   task_pid_nr(current),
   (long long)ktime_to_ns(delta) >> 10);
 }
}
EXPORT_SYMBOL_GPL(async_synchronize_cookie_domain);
void async_synchronize_cookie(async_cookie_t cookie)
{
 async_synchronize_cookie_domain(cookie, &async_dfl_domain);
}
EXPORT_SYMBOL_GPL(async_synchronize_cookie);






bool current_is_async(void)
{
 struct worker *worker = current_wq_worker();

 return worker && worker->current_func == async_run_entry_fn;
}
EXPORT_SYMBOL_GPL(current_is_async);







static int audit_initialized;

u32 audit_enabled;
u32 audit_ever_enabled;

EXPORT_SYMBOL_GPL(audit_enabled);


static u32 audit_default;


static u32 audit_failure = AUDIT_FAIL_PRINTK;






int audit_pid;
static __u32 audit_nlk_portid;




static u32 audit_rate_limit;



static u32 audit_backlog_limit = 64;
static u32 audit_backlog_wait_time_master = AUDIT_BACKLOG_WAIT_TIME;
static u32 audit_backlog_wait_time = AUDIT_BACKLOG_WAIT_TIME;


kuid_t audit_sig_uid = INVALID_UID;
pid_t audit_sig_pid = -1;
u32 audit_sig_sid = 0;
static atomic_t audit_lost = ATOMIC_INIT(0);


static struct sock *audit_sock;
static int audit_net_id;


struct list_head audit_inode_hash[AUDIT_INODE_BUCKETS];




static DEFINE_SPINLOCK(audit_freelist_lock);
static int audit_freelist_count;
static LIST_HEAD(audit_freelist);

static struct sk_buff_head audit_skb_queue;

static struct sk_buff_head audit_skb_hold_queue;
static struct task_struct *kauditd_task;
static DECLARE_WAIT_QUEUE_HEAD(kauditd_wait);
static DECLARE_WAIT_QUEUE_HEAD(audit_backlog_wait);

static struct audit_features af = {.vers = AUDIT_FEATURE_VERSION,
       .mask = -1,
       .features = 0,
       .lock = 0,};

static char *audit_feature_names[2] = {
 "only_unset_loginuid",
 "loginuid_immutable",
};



DEFINE_MUTEX(audit_cmd_mutex);













struct audit_buffer {
 struct list_head list;
 struct sk_buff *skb;
 struct audit_context *ctx;
 gfp_t gfp_mask;
};

struct audit_reply {
 __u32 portid;
 struct net *net;
 struct sk_buff *skb;
};

static void audit_set_portid(struct audit_buffer *ab, __u32 portid)
{
 if (ab) {
  struct nlmsghdr *nlh = nlmsg_hdr(ab->skb);
  nlh->nlmsg_pid = portid;
 }
}

void audit_panic(const char *message)
{
 switch (audit_failure) {
 case AUDIT_FAIL_SILENT:
  break;
 case AUDIT_FAIL_PRINTK:
  if (printk_ratelimit())
   pr_err("%s\n", message);
  break;
 case AUDIT_FAIL_PANIC:

  if (audit_pid)
   panic("audit: %s\n", message);
  break;
 }
}

static inline int audit_rate_check(void)
{
 static unsigned long last_check = 0;
 static int messages = 0;
 static DEFINE_SPINLOCK(lock);
 unsigned long flags;
 unsigned long now;
 unsigned long elapsed;
 int retval = 0;

 if (!audit_rate_limit) return 1;

 spin_lock_irqsave(&lock, flags);
 if (++messages < audit_rate_limit) {
  retval = 1;
 } else {
  now = jiffies;
  elapsed = now - last_check;
  if (elapsed > HZ) {
   last_check = now;
   messages = 0;
   retval = 1;
  }
 }
 spin_unlock_irqrestore(&lock, flags);

 return retval;
}
void audit_log_lost(const char *message)
{
 static unsigned long last_msg = 0;
 static DEFINE_SPINLOCK(lock);
 unsigned long flags;
 unsigned long now;
 int print;

 atomic_inc(&audit_lost);

 print = (audit_failure == AUDIT_FAIL_PANIC || !audit_rate_limit);

 if (!print) {
  spin_lock_irqsave(&lock, flags);
  now = jiffies;
  if (now - last_msg > HZ) {
   print = 1;
   last_msg = now;
  }
  spin_unlock_irqrestore(&lock, flags);
 }

 if (print) {
  if (printk_ratelimit())
   pr_warn("audit_lost=%u audit_rate_limit=%u audit_backlog_limit=%u\n",
    atomic_read(&audit_lost),
    audit_rate_limit,
    audit_backlog_limit);
  audit_panic(message);
 }
}

static int audit_log_config_change(char *function_name, u32 new, u32 old,
       int allow_changes)
{
 struct audit_buffer *ab;
 int rc = 0;

 ab = audit_log_start(NULL, GFP_KERNEL, AUDIT_CONFIG_CHANGE);
 if (unlikely(!ab))
  return rc;
 audit_log_format(ab, "%s=%u old=%u", function_name, new, old);
 audit_log_session_info(ab);
 rc = audit_log_task_context(ab);
 if (rc)
  allow_changes = 0;
 audit_log_format(ab, " res=%d", allow_changes);
 audit_log_end(ab);
 return rc;
}

static int audit_do_config_change(char *function_name, u32 *to_change, u32 new)
{
 int allow_changes, rc = 0;
 u32 old = *to_change;


 if (audit_enabled == AUDIT_LOCKED)
  allow_changes = 0;
 else
  allow_changes = 1;

 if (audit_enabled != AUDIT_OFF) {
  rc = audit_log_config_change(function_name, new, old, allow_changes);
  if (rc)
   allow_changes = 0;
 }


 if (allow_changes == 1)
  *to_change = new;

 else if (rc == 0)
  rc = -EPERM;
 return rc;
}

static int audit_set_rate_limit(u32 limit)
{
 return audit_do_config_change("audit_rate_limit", &audit_rate_limit, limit);
}

static int audit_set_backlog_limit(u32 limit)
{
 return audit_do_config_change("audit_backlog_limit", &audit_backlog_limit, limit);
}

static int audit_set_backlog_wait_time(u32 timeout)
{
 return audit_do_config_change("audit_backlog_wait_time",
          &audit_backlog_wait_time_master, timeout);
}

static int audit_set_enabled(u32 state)
{
 int rc;
 if (state > AUDIT_LOCKED)
  return -EINVAL;

 rc = audit_do_config_change("audit_enabled", &audit_enabled, state);
 if (!rc)
  audit_ever_enabled |= !!state;

 return rc;
}

static int audit_set_failure(u32 state)
{
 if (state != AUDIT_FAIL_SILENT
     && state != AUDIT_FAIL_PRINTK
     && state != AUDIT_FAIL_PANIC)
  return -EINVAL;

 return audit_do_config_change("audit_failure", &audit_failure, state);
}
static void audit_hold_skb(struct sk_buff *skb)
{
 if (audit_default &&
     (!audit_backlog_limit ||
      skb_queue_len(&audit_skb_hold_queue) < audit_backlog_limit))
  skb_queue_tail(&audit_skb_hold_queue, skb);
 else
  kfree_skb(skb);
}





static void audit_printk_skb(struct sk_buff *skb)
{
 struct nlmsghdr *nlh = nlmsg_hdr(skb);
 char *data = nlmsg_data(nlh);

 if (nlh->nlmsg_type != AUDIT_EOE) {
  if (printk_ratelimit())
   pr_notice("type=%d %s\n", nlh->nlmsg_type, data);
  else
   audit_log_lost("printk limit exceeded");
 }

 audit_hold_skb(skb);
}

static void kauditd_send_skb(struct sk_buff *skb)
{
 int err;
 int attempts = 0;

restart:

 skb_get(skb);
 err = netlink_unicast(audit_sock, skb, audit_nlk_portid, 0);
 if (err < 0) {
  pr_err("netlink_unicast sending to audit_pid=%d returned error: %d\n",
         audit_pid, err);
  if (audit_pid) {
   if (err == -ECONNREFUSED || err == -EPERM
       || ++attempts >= AUDITD_RETRIES) {
    char s[32];

    snprintf(s, sizeof(s), "audit_pid=%d reset", audit_pid);
    audit_log_lost(s);
    audit_pid = 0;
    audit_sock = NULL;
   } else {
    pr_warn("re-scheduling(#%d) write to audit_pid=%d\n",
     attempts, audit_pid);
    set_current_state(TASK_INTERRUPTIBLE);
    schedule();
    goto restart;
   }
  }

  audit_hold_skb(skb);
 } else

  consume_skb(skb);
}







static void kauditd_send_multicast_skb(struct sk_buff *skb, gfp_t gfp_mask)
{
 struct sk_buff *copy;
 struct audit_net *aunet = net_generic(&init_net, audit_net_id);
 struct sock *sock = aunet->nlsk;

 if (!netlink_has_listeners(sock, AUDIT_NLGRP_READLOG))
  return;
 copy = skb_copy(skb, gfp_mask);
 if (!copy)
  return;

 nlmsg_multicast(sock, copy, 0, AUDIT_NLGRP_READLOG, gfp_mask);
}
static void flush_hold_queue(void)
{
 struct sk_buff *skb;

 if (!audit_default || !audit_pid)
  return;

 skb = skb_dequeue(&audit_skb_hold_queue);
 if (likely(!skb))
  return;

 while (skb && audit_pid) {
  kauditd_send_skb(skb);
  skb = skb_dequeue(&audit_skb_hold_queue);
 }





 consume_skb(skb);
}

static int kauditd_thread(void *dummy)
{
 set_freezable();
 while (!kthread_should_stop()) {
  struct sk_buff *skb;

  flush_hold_queue();

  skb = skb_dequeue(&audit_skb_queue);

  if (skb) {
   if (!audit_backlog_limit ||
       (skb_queue_len(&audit_skb_queue) <= audit_backlog_limit))
    wake_up(&audit_backlog_wait);
   if (audit_pid)
    kauditd_send_skb(skb);
   else
    audit_printk_skb(skb);
   continue;
  }

  wait_event_freezable(kauditd_wait, skb_queue_len(&audit_skb_queue));
 }
 return 0;
}

int audit_send_list(void *_dest)
{
 struct audit_netlink_list *dest = _dest;
 struct sk_buff *skb;
 struct net *net = dest->net;
 struct audit_net *aunet = net_generic(net, audit_net_id);


 mutex_lock(&audit_cmd_mutex);
 mutex_unlock(&audit_cmd_mutex);

 while ((skb = __skb_dequeue(&dest->q)) != NULL)
  netlink_unicast(aunet->nlsk, skb, dest->portid, 0);

 put_net(net);
 kfree(dest);

 return 0;
}

struct sk_buff *audit_make_reply(__u32 portid, int seq, int type, int done,
     int multi, const void *payload, int size)
{
 struct sk_buff *skb;
 struct nlmsghdr *nlh;
 void *data;
 int flags = multi ? NLM_F_MULTI : 0;
 int t = done ? NLMSG_DONE : type;

 skb = nlmsg_new(size, GFP_KERNEL);
 if (!skb)
  return NULL;

 nlh = nlmsg_put(skb, portid, seq, t, size, flags);
 if (!nlh)
  goto out_kfree_skb;
 data = nlmsg_data(nlh);
 memcpy(data, payload, size);
 return skb;

out_kfree_skb:
 kfree_skb(skb);
 return NULL;
}

static int audit_send_reply_thread(void *arg)
{
 struct audit_reply *reply = (struct audit_reply *)arg;
 struct net *net = reply->net;
 struct audit_net *aunet = net_generic(net, audit_net_id);

 mutex_lock(&audit_cmd_mutex);
 mutex_unlock(&audit_cmd_mutex);



 netlink_unicast(aunet->nlsk , reply->skb, reply->portid, 0);
 put_net(net);
 kfree(reply);
 return 0;
}
static void audit_send_reply(struct sk_buff *request_skb, int seq, int type, int done,
        int multi, const void *payload, int size)
{
 u32 portid = NETLINK_CB(request_skb).portid;
 struct net *net = sock_net(NETLINK_CB(request_skb).sk);
 struct sk_buff *skb;
 struct task_struct *tsk;
 struct audit_reply *reply = kmalloc(sizeof(struct audit_reply),
         GFP_KERNEL);

 if (!reply)
  return;

 skb = audit_make_reply(portid, seq, type, done, multi, payload, size);
 if (!skb)
  goto out;

 reply->net = get_net(net);
 reply->portid = portid;
 reply->skb = skb;

 tsk = kthread_run(audit_send_reply_thread, reply, "audit_send_reply");
 if (!IS_ERR(tsk))
  return;
 kfree_skb(skb);
out:
 kfree(reply);
}





static int audit_netlink_ok(struct sk_buff *skb, u16 msg_type)
{
 int err = 0;
 if (current_user_ns() != &init_user_ns)
  return -ECONNREFUSED;

 switch (msg_type) {
 case AUDIT_LIST:
 case AUDIT_ADD:
 case AUDIT_DEL:
  return -EOPNOTSUPP;
 case AUDIT_GET:
 case AUDIT_SET:
 case AUDIT_GET_FEATURE:
 case AUDIT_SET_FEATURE:
 case AUDIT_LIST_RULES:
 case AUDIT_ADD_RULE:
 case AUDIT_DEL_RULE:
 case AUDIT_SIGNAL_INFO:
 case AUDIT_TTY_GET:
 case AUDIT_TTY_SET:
 case AUDIT_TRIM:
 case AUDIT_MAKE_EQUIV:


  if (task_active_pid_ns(current) != &init_pid_ns)
   return -EPERM;

  if (!netlink_capable(skb, CAP_AUDIT_CONTROL))
   err = -EPERM;
  break;
 case AUDIT_USER:
 case AUDIT_FIRST_USER_MSG ... AUDIT_LAST_USER_MSG:
 case AUDIT_FIRST_USER_MSG2 ... AUDIT_LAST_USER_MSG2:
  if (!netlink_capable(skb, CAP_AUDIT_WRITE))
   err = -EPERM;
  break;
 default:
  err = -EINVAL;
 }

 return err;
}

static void audit_log_common_recv_msg(struct audit_buffer **ab, u16 msg_type)
{
 uid_t uid = from_kuid(&init_user_ns, current_uid());
 pid_t pid = task_tgid_nr(current);

 if (!audit_enabled && msg_type != AUDIT_USER_AVC) {
  *ab = NULL;
  return;
 }

 *ab = audit_log_start(NULL, GFP_KERNEL, msg_type);
 if (unlikely(!*ab))
  return;
 audit_log_format(*ab, "pid=%d uid=%u", pid, uid);
 audit_log_session_info(*ab);
 audit_log_task_context(*ab);
}

int is_audit_feature_set(int i)
{
 return af.features & AUDIT_FEATURE_TO_MASK(i);
}


static int audit_get_feature(struct sk_buff *skb)
{
 u32 seq;

 seq = nlmsg_hdr(skb)->nlmsg_seq;

 audit_send_reply(skb, seq, AUDIT_GET_FEATURE, 0, 0, &af, sizeof(af));

 return 0;
}

static void audit_log_feature_change(int which, u32 old_feature, u32 new_feature,
         u32 old_lock, u32 new_lock, int res)
{
 struct audit_buffer *ab;

 if (audit_enabled == AUDIT_OFF)
  return;

 ab = audit_log_start(NULL, GFP_KERNEL, AUDIT_FEATURE_CHANGE);
 audit_log_task_info(ab, current);
 audit_log_format(ab, " feature=%s old=%u new=%u old_lock=%u new_lock=%u res=%d",
    audit_feature_names[which], !!old_feature, !!new_feature,
    !!old_lock, !!new_lock, res);
 audit_log_end(ab);
}

static int audit_set_feature(struct sk_buff *skb)
{
 struct audit_features *uaf;
 int i;

 BUILD_BUG_ON(AUDIT_LAST_FEATURE + 1 > ARRAY_SIZE(audit_feature_names));
 uaf = nlmsg_data(nlmsg_hdr(skb));



 for (i = 0; i <= AUDIT_LAST_FEATURE; i++) {
  u32 feature = AUDIT_FEATURE_TO_MASK(i);
  u32 old_feature, new_feature, old_lock, new_lock;


  if (!(feature & uaf->mask))
   continue;

  old_feature = af.features & feature;
  new_feature = uaf->features & feature;
  new_lock = (uaf->lock | af.lock) & feature;
  old_lock = af.lock & feature;


  if (old_lock && (new_feature != old_feature)) {
   audit_log_feature_change(i, old_feature, new_feature,
       old_lock, new_lock, 0);
   return -EPERM;
  }
 }

 for (i = 0; i <= AUDIT_LAST_FEATURE; i++) {
  u32 feature = AUDIT_FEATURE_TO_MASK(i);
  u32 old_feature, new_feature, old_lock, new_lock;


  if (!(feature & uaf->mask))
   continue;

  old_feature = af.features & feature;
  new_feature = uaf->features & feature;
  old_lock = af.lock & feature;
  new_lock = (uaf->lock | af.lock) & feature;

  if (new_feature != old_feature)
   audit_log_feature_change(i, old_feature, new_feature,
       old_lock, new_lock, 1);

  if (new_feature)
   af.features |= feature;
  else
   af.features &= ~feature;
  af.lock |= new_lock;
 }

 return 0;
}

static int audit_replace(pid_t pid)
{
 struct sk_buff *skb = audit_make_reply(0, 0, AUDIT_REPLACE, 0, 0,
            &pid, sizeof(pid));

 if (!skb)
  return -ENOMEM;
 return netlink_unicast(audit_sock, skb, audit_nlk_portid, 0);
}

static int audit_receive_msg(struct sk_buff *skb, struct nlmsghdr *nlh)
{
 u32 seq;
 void *data;
 int err;
 struct audit_buffer *ab;
 u16 msg_type = nlh->nlmsg_type;
 struct audit_sig_info *sig_data;
 char *ctx = NULL;
 u32 len;

 err = audit_netlink_ok(skb, msg_type);
 if (err)
  return err;



 if (!kauditd_task) {
  kauditd_task = kthread_run(kauditd_thread, NULL, "kauditd");
  if (IS_ERR(kauditd_task)) {
   err = PTR_ERR(kauditd_task);
   kauditd_task = NULL;
   return err;
  }
 }
 seq = nlh->nlmsg_seq;
 data = nlmsg_data(nlh);

 switch (msg_type) {
 case AUDIT_GET: {
  struct audit_status s;
  memset(&s, 0, sizeof(s));
  s.enabled = audit_enabled;
  s.failure = audit_failure;
  s.pid = audit_pid;
  s.rate_limit = audit_rate_limit;
  s.backlog_limit = audit_backlog_limit;
  s.lost = atomic_read(&audit_lost);
  s.backlog = skb_queue_len(&audit_skb_queue);
  s.feature_bitmap = AUDIT_FEATURE_BITMAP_ALL;
  s.backlog_wait_time = audit_backlog_wait_time_master;
  audit_send_reply(skb, seq, AUDIT_GET, 0, 0, &s, sizeof(s));
  break;
 }
 case AUDIT_SET: {
  struct audit_status s;
  memset(&s, 0, sizeof(s));

  memcpy(&s, data, min_t(size_t, sizeof(s), nlmsg_len(nlh)));
  if (s.mask & AUDIT_STATUS_ENABLED) {
   err = audit_set_enabled(s.enabled);
   if (err < 0)
    return err;
  }
  if (s.mask & AUDIT_STATUS_FAILURE) {
   err = audit_set_failure(s.failure);
   if (err < 0)
    return err;
  }
  if (s.mask & AUDIT_STATUS_PID) {
   int new_pid = s.pid;
   pid_t requesting_pid = task_tgid_vnr(current);

   if ((!new_pid) && (requesting_pid != audit_pid)) {
    audit_log_config_change("audit_pid", new_pid, audit_pid, 0);
    return -EACCES;
   }
   if (audit_pid && new_pid &&
       audit_replace(requesting_pid) != -ECONNREFUSED) {
    audit_log_config_change("audit_pid", new_pid, audit_pid, 0);
    return -EEXIST;
   }
   if (audit_enabled != AUDIT_OFF)
    audit_log_config_change("audit_pid", new_pid, audit_pid, 1);
   audit_pid = new_pid;
   audit_nlk_portid = NETLINK_CB(skb).portid;
   audit_sock = skb->sk;
  }
  if (s.mask & AUDIT_STATUS_RATE_LIMIT) {
   err = audit_set_rate_limit(s.rate_limit);
   if (err < 0)
    return err;
  }
  if (s.mask & AUDIT_STATUS_BACKLOG_LIMIT) {
   err = audit_set_backlog_limit(s.backlog_limit);
   if (err < 0)
    return err;
  }
  if (s.mask & AUDIT_STATUS_BACKLOG_WAIT_TIME) {
   if (sizeof(s) > (size_t)nlh->nlmsg_len)
    return -EINVAL;
   if (s.backlog_wait_time > 10*AUDIT_BACKLOG_WAIT_TIME)
    return -EINVAL;
   err = audit_set_backlog_wait_time(s.backlog_wait_time);
   if (err < 0)
    return err;
  }
  break;
 }
 case AUDIT_GET_FEATURE:
  err = audit_get_feature(skb);
  if (err)
   return err;
  break;
 case AUDIT_SET_FEATURE:
  err = audit_set_feature(skb);
  if (err)
   return err;
  break;
 case AUDIT_USER:
 case AUDIT_FIRST_USER_MSG ... AUDIT_LAST_USER_MSG:
 case AUDIT_FIRST_USER_MSG2 ... AUDIT_LAST_USER_MSG2:
  if (!audit_enabled && msg_type != AUDIT_USER_AVC)
   return 0;

  err = audit_filter_user(msg_type);
  if (err == 1) {
   err = 0;
   if (msg_type == AUDIT_USER_TTY) {
    err = tty_audit_push();
    if (err)
     break;
   }
   mutex_unlock(&audit_cmd_mutex);
   audit_log_common_recv_msg(&ab, msg_type);
   if (msg_type != AUDIT_USER_TTY)
    audit_log_format(ab, " msg='%.*s'",
       AUDIT_MESSAGE_TEXT_MAX,
       (char *)data);
   else {
    int size;

    audit_log_format(ab, " data=");
    size = nlmsg_len(nlh);
    if (size > 0 &&
        ((unsigned char *)data)[size - 1] == '\0')
     size--;
    audit_log_n_untrustedstring(ab, data, size);
   }
   audit_set_portid(ab, NETLINK_CB(skb).portid);
   audit_log_end(ab);
   mutex_lock(&audit_cmd_mutex);
  }
  break;
 case AUDIT_ADD_RULE:
 case AUDIT_DEL_RULE:
  if (nlmsg_len(nlh) < sizeof(struct audit_rule_data))
   return -EINVAL;
  if (audit_enabled == AUDIT_LOCKED) {
   audit_log_common_recv_msg(&ab, AUDIT_CONFIG_CHANGE);
   audit_log_format(ab, " audit_enabled=%d res=0", audit_enabled);
   audit_log_end(ab);
   return -EPERM;
  }
  err = audit_rule_change(msg_type, NETLINK_CB(skb).portid,
        seq, data, nlmsg_len(nlh));
  break;
 case AUDIT_LIST_RULES:
  err = audit_list_rules_send(skb, seq);
  break;
 case AUDIT_TRIM:
  audit_trim_trees();
  audit_log_common_recv_msg(&ab, AUDIT_CONFIG_CHANGE);
  audit_log_format(ab, " op=trim res=1");
  audit_log_end(ab);
  break;
 case AUDIT_MAKE_EQUIV: {
  void *bufp = data;
  u32 sizes[2];
  size_t msglen = nlmsg_len(nlh);
  char *old, *new;

  err = -EINVAL;
  if (msglen < 2 * sizeof(u32))
   break;
  memcpy(sizes, bufp, 2 * sizeof(u32));
  bufp += 2 * sizeof(u32);
  msglen -= 2 * sizeof(u32);
  old = audit_unpack_string(&bufp, &msglen, sizes[0]);
  if (IS_ERR(old)) {
   err = PTR_ERR(old);
   break;
  }
  new = audit_unpack_string(&bufp, &msglen, sizes[1]);
  if (IS_ERR(new)) {
   err = PTR_ERR(new);
   kfree(old);
   break;
  }

  err = audit_tag_tree(old, new);

  audit_log_common_recv_msg(&ab, AUDIT_CONFIG_CHANGE);

  audit_log_format(ab, " op=make_equiv old=");
  audit_log_untrustedstring(ab, old);
  audit_log_format(ab, " new=");
  audit_log_untrustedstring(ab, new);
  audit_log_format(ab, " res=%d", !err);
  audit_log_end(ab);
  kfree(old);
  kfree(new);
  break;
 }
 case AUDIT_SIGNAL_INFO:
  len = 0;
  if (audit_sig_sid) {
   err = security_secid_to_secctx(audit_sig_sid, &ctx, &len);
   if (err)
    return err;
  }
  sig_data = kmalloc(sizeof(*sig_data) + len, GFP_KERNEL);
  if (!sig_data) {
   if (audit_sig_sid)
    security_release_secctx(ctx, len);
   return -ENOMEM;
  }
  sig_data->uid = from_kuid(&init_user_ns, audit_sig_uid);
  sig_data->pid = audit_sig_pid;
  if (audit_sig_sid) {
   memcpy(sig_data->ctx, ctx, len);
   security_release_secctx(ctx, len);
  }
  audit_send_reply(skb, seq, AUDIT_SIGNAL_INFO, 0, 0,
     sig_data, sizeof(*sig_data) + len);
  kfree(sig_data);
  break;
 case AUDIT_TTY_GET: {
  struct audit_tty_status s;
  unsigned int t;

  t = READ_ONCE(current->signal->audit_tty);
  s.enabled = t & AUDIT_TTY_ENABLE;
  s.log_passwd = !!(t & AUDIT_TTY_LOG_PASSWD);

  audit_send_reply(skb, seq, AUDIT_TTY_GET, 0, 0, &s, sizeof(s));
  break;
 }
 case AUDIT_TTY_SET: {
  struct audit_tty_status s, old;
  struct audit_buffer *ab;
  unsigned int t;

  memset(&s, 0, sizeof(s));

  memcpy(&s, data, min_t(size_t, sizeof(s), nlmsg_len(nlh)));

  if ((s.enabled != 0 && s.enabled != 1) ||
      (s.log_passwd != 0 && s.log_passwd != 1))
   err = -EINVAL;

  if (err)
   t = READ_ONCE(current->signal->audit_tty);
  else {
   t = s.enabled | (-s.log_passwd & AUDIT_TTY_LOG_PASSWD);
   t = xchg(&current->signal->audit_tty, t);
  }
  old.enabled = t & AUDIT_TTY_ENABLE;
  old.log_passwd = !!(t & AUDIT_TTY_LOG_PASSWD);

  audit_log_common_recv_msg(&ab, AUDIT_CONFIG_CHANGE);
  audit_log_format(ab, " op=tty_set old-enabled=%d new-enabled=%d"
     " old-log_passwd=%d new-log_passwd=%d res=%d",
     old.enabled, s.enabled, old.log_passwd,
     s.log_passwd, !err);
  audit_log_end(ab);
  break;
 }
 default:
  err = -EINVAL;
  break;
 }

 return err < 0 ? err : 0;
}





static void audit_receive_skb(struct sk_buff *skb)
{
 struct nlmsghdr *nlh;




 int len;
 int err;

 nlh = nlmsg_hdr(skb);
 len = skb->len;

 while (nlmsg_ok(nlh, len)) {
  err = audit_receive_msg(skb, nlh);

  if (err || (nlh->nlmsg_flags & NLM_F_ACK))
   netlink_ack(skb, nlh, err);

  nlh = nlmsg_next(nlh, &len);
 }
}


static void audit_receive(struct sk_buff *skb)
{
 mutex_lock(&audit_cmd_mutex);
 audit_receive_skb(skb);
 mutex_unlock(&audit_cmd_mutex);
}


static int audit_bind(struct net *net, int group)
{
 if (!capable(CAP_AUDIT_READ))
  return -EPERM;

 return 0;
}

static int __net_init audit_net_init(struct net *net)
{
 struct netlink_kernel_cfg cfg = {
  .input = audit_receive,
  .bind = audit_bind,
  .flags = NL_CFG_F_NONROOT_RECV,
  .groups = AUDIT_NLGRP_MAX,
 };

 struct audit_net *aunet = net_generic(net, audit_net_id);

 aunet->nlsk = netlink_kernel_create(net, NETLINK_AUDIT, &cfg);
 if (aunet->nlsk == NULL) {
  audit_panic("cannot initialize netlink socket in namespace");
  return -ENOMEM;
 }
 aunet->nlsk->sk_sndtimeo = MAX_SCHEDULE_TIMEOUT;
 return 0;
}

static void __net_exit audit_net_exit(struct net *net)
{
 struct audit_net *aunet = net_generic(net, audit_net_id);
 struct sock *sock = aunet->nlsk;
 if (sock == audit_sock) {
  audit_pid = 0;
  audit_sock = NULL;
 }

 RCU_INIT_POINTER(aunet->nlsk, NULL);
 synchronize_net();
 netlink_kernel_release(sock);
}

static struct pernet_operations audit_net_ops __net_initdata = {
 .init = audit_net_init,
 .exit = audit_net_exit,
 .id = &audit_net_id,
 .size = sizeof(struct audit_net),
};


static int __init audit_init(void)
{
 int i;

 if (audit_initialized == AUDIT_DISABLED)
  return 0;

 pr_info("initializing netlink subsys (%s)\n",
  audit_default ? "enabled" : "disabled");
 register_pernet_subsys(&audit_net_ops);

 skb_queue_head_init(&audit_skb_queue);
 skb_queue_head_init(&audit_skb_hold_queue);
 audit_initialized = AUDIT_INITIALIZED;
 audit_enabled = audit_default;
 audit_ever_enabled |= !!audit_default;

 audit_log(NULL, GFP_KERNEL, AUDIT_KERNEL, "initialized");

 for (i = 0; i < AUDIT_INODE_BUCKETS; i++)
  INIT_LIST_HEAD(&audit_inode_hash[i]);

 return 0;
}
__initcall(audit_init);


static int __init audit_enable(char *str)
{
 audit_default = !!simple_strtol(str, NULL, 0);
 if (!audit_default)
  audit_initialized = AUDIT_DISABLED;

 pr_info("%s\n", audit_default ?
  "enabled (after initialization)" : "disabled (until reboot)");

 return 1;
}
__setup("audit=", audit_enable);



static int __init audit_backlog_limit_set(char *str)
{
 u32 audit_backlog_limit_arg;

 pr_info("audit_backlog_limit: ");
 if (kstrtouint(str, 0, &audit_backlog_limit_arg)) {
  pr_cont("using default of %u, unable to parse %s\n",
   audit_backlog_limit, str);
  return 1;
 }

 audit_backlog_limit = audit_backlog_limit_arg;
 pr_cont("%d\n", audit_backlog_limit);

 return 1;
}
__setup("audit_backlog_limit=", audit_backlog_limit_set);

static void audit_buffer_free(struct audit_buffer *ab)
{
 unsigned long flags;

 if (!ab)
  return;

 kfree_skb(ab->skb);
 spin_lock_irqsave(&audit_freelist_lock, flags);
 if (audit_freelist_count > AUDIT_MAXFREE)
  kfree(ab);
 else {
  audit_freelist_count++;
  list_add(&ab->list, &audit_freelist);
 }
 spin_unlock_irqrestore(&audit_freelist_lock, flags);
}

static struct audit_buffer * audit_buffer_alloc(struct audit_context *ctx,
      gfp_t gfp_mask, int type)
{
 unsigned long flags;
 struct audit_buffer *ab = NULL;
 struct nlmsghdr *nlh;

 spin_lock_irqsave(&audit_freelist_lock, flags);
 if (!list_empty(&audit_freelist)) {
  ab = list_entry(audit_freelist.next,
    struct audit_buffer, list);
  list_del(&ab->list);
  --audit_freelist_count;
 }
 spin_unlock_irqrestore(&audit_freelist_lock, flags);

 if (!ab) {
  ab = kmalloc(sizeof(*ab), gfp_mask);
  if (!ab)
   goto err;
 }

 ab->ctx = ctx;
 ab->gfp_mask = gfp_mask;

 ab->skb = nlmsg_new(AUDIT_BUFSIZ, gfp_mask);
 if (!ab->skb)
  goto err;

 nlh = nlmsg_put(ab->skb, 0, 0, type, 0, 0);
 if (!nlh)
  goto out_kfree_skb;

 return ab;

out_kfree_skb:
 kfree_skb(ab->skb);
 ab->skb = NULL;
err:
 audit_buffer_free(ab);
 return NULL;
}
unsigned int audit_serial(void)
{
 static atomic_t serial = ATOMIC_INIT(0);

 return atomic_add_return(1, &serial);
}

static inline void audit_get_stamp(struct audit_context *ctx,
       struct timespec *t, unsigned int *serial)
{
 if (!ctx || !auditsc_get_stamp(ctx, t, serial)) {
  *t = CURRENT_TIME;
  *serial = audit_serial();
 }
}




static long wait_for_auditd(long sleep_time)
{
 DECLARE_WAITQUEUE(wait, current);

 if (audit_backlog_limit &&
     skb_queue_len(&audit_skb_queue) > audit_backlog_limit) {
  add_wait_queue_exclusive(&audit_backlog_wait, &wait);
  set_current_state(TASK_UNINTERRUPTIBLE);
  sleep_time = schedule_timeout(sleep_time);
  remove_wait_queue(&audit_backlog_wait, &wait);
 }

 return sleep_time;
}
struct audit_buffer *audit_log_start(struct audit_context *ctx, gfp_t gfp_mask,
         int type)
{
 struct audit_buffer *ab = NULL;
 struct timespec t;
 unsigned int uninitialized_var(serial);
 int reserve = 5;

 unsigned long timeout_start = jiffies;

 if (audit_initialized != AUDIT_INITIALIZED)
  return NULL;

 if (unlikely(audit_filter_type(type)))
  return NULL;

 if (gfp_mask & __GFP_DIRECT_RECLAIM) {
  if (audit_pid && audit_pid == current->tgid)
   gfp_mask &= ~__GFP_DIRECT_RECLAIM;
  else
   reserve = 0;
 }

 while (audit_backlog_limit
        && skb_queue_len(&audit_skb_queue) > audit_backlog_limit + reserve) {
  if (gfp_mask & __GFP_DIRECT_RECLAIM && audit_backlog_wait_time) {
   long sleep_time;

   sleep_time = timeout_start + audit_backlog_wait_time - jiffies;
   if (sleep_time > 0) {
    sleep_time = wait_for_auditd(sleep_time);
    if (sleep_time > 0)
     continue;
   }
  }
  if (audit_rate_check() && printk_ratelimit())
   pr_warn("audit_backlog=%d > audit_backlog_limit=%d\n",
    skb_queue_len(&audit_skb_queue),
    audit_backlog_limit);
  audit_log_lost("backlog limit exceeded");
  audit_backlog_wait_time = 0;
  wake_up(&audit_backlog_wait);
  return NULL;
 }

 if (!reserve && !audit_backlog_wait_time)
  audit_backlog_wait_time = audit_backlog_wait_time_master;

 ab = audit_buffer_alloc(ctx, gfp_mask, type);
 if (!ab) {
  audit_log_lost("out of memory in audit_log_start");
  return NULL;
 }

 audit_get_stamp(ab->ctx, &t, &serial);

 audit_log_format(ab, "audit(%lu.%03lu:%u): ",
    t.tv_sec, t.tv_nsec/1000000, serial);
 return ab;
}
static inline int audit_expand(struct audit_buffer *ab, int extra)
{
 struct sk_buff *skb = ab->skb;
 int oldtail = skb_tailroom(skb);
 int ret = pskb_expand_head(skb, 0, extra, ab->gfp_mask);
 int newtail = skb_tailroom(skb);

 if (ret < 0) {
  audit_log_lost("out of memory in audit_expand");
  return 0;
 }

 skb->truesize += newtail - oldtail;
 return newtail;
}







static void audit_log_vformat(struct audit_buffer *ab, const char *fmt,
         va_list args)
{
 int len, avail;
 struct sk_buff *skb;
 va_list args2;

 if (!ab)
  return;

 BUG_ON(!ab->skb);
 skb = ab->skb;
 avail = skb_tailroom(skb);
 if (avail == 0) {
  avail = audit_expand(ab, AUDIT_BUFSIZ);
  if (!avail)
   goto out;
 }
 va_copy(args2, args);
 len = vsnprintf(skb_tail_pointer(skb), avail, fmt, args);
 if (len >= avail) {



  avail = audit_expand(ab,
   max_t(unsigned, AUDIT_BUFSIZ, 1+len-avail));
  if (!avail)
   goto out_va_end;
  len = vsnprintf(skb_tail_pointer(skb), avail, fmt, args2);
 }
 if (len > 0)
  skb_put(skb, len);
out_va_end:
 va_end(args2);
out:
 return;
}
void audit_log_format(struct audit_buffer *ab, const char *fmt, ...)
{
 va_list args;

 if (!ab)
  return;
 va_start(args, fmt);
 audit_log_vformat(ab, fmt, args);
 va_end(args);
}
void audit_log_n_hex(struct audit_buffer *ab, const unsigned char *buf,
  size_t len)
{
 int i, avail, new_len;
 unsigned char *ptr;
 struct sk_buff *skb;

 if (!ab)
  return;

 BUG_ON(!ab->skb);
 skb = ab->skb;
 avail = skb_tailroom(skb);
 new_len = len<<1;
 if (new_len >= avail) {

  new_len = AUDIT_BUFSIZ*(((new_len-avail)/AUDIT_BUFSIZ) + 1);
  avail = audit_expand(ab, new_len);
  if (!avail)
   return;
 }

 ptr = skb_tail_pointer(skb);
 for (i = 0; i < len; i++)
  ptr = hex_byte_pack_upper(ptr, buf[i]);
 *ptr = 0;
 skb_put(skb, len << 1);
}





void audit_log_n_string(struct audit_buffer *ab, const char *string,
   size_t slen)
{
 int avail, new_len;
 unsigned char *ptr;
 struct sk_buff *skb;

 if (!ab)
  return;

 BUG_ON(!ab->skb);
 skb = ab->skb;
 avail = skb_tailroom(skb);
 new_len = slen + 3;
 if (new_len > avail) {
  avail = audit_expand(ab, new_len);
  if (!avail)
   return;
 }
 ptr = skb_tail_pointer(skb);
 *ptr++ = '"';
 memcpy(ptr, string, slen);
 ptr += slen;
 *ptr++ = '"';
 *ptr = 0;
 skb_put(skb, slen + 2);
}






bool audit_string_contains_control(const char *string, size_t len)
{
 const unsigned char *p;
 for (p = string; p < (const unsigned char *)string + len; p++) {
  if (*p == '"' || *p < 0x21 || *p > 0x7e)
   return true;
 }
 return false;
}
void audit_log_n_untrustedstring(struct audit_buffer *ab, const char *string,
     size_t len)
{
 if (audit_string_contains_control(string, len))
  audit_log_n_hex(ab, string, len);
 else
  audit_log_n_string(ab, string, len);
}
void audit_log_untrustedstring(struct audit_buffer *ab, const char *string)
{
 audit_log_n_untrustedstring(ab, string, strlen(string));
}


void audit_log_d_path(struct audit_buffer *ab, const char *prefix,
        const struct path *path)
{
 char *p, *pathname;

 if (prefix)
  audit_log_format(ab, "%s", prefix);


 pathname = kmalloc(PATH_MAX+11, ab->gfp_mask);
 if (!pathname) {
  audit_log_string(ab, "<no_memory>");
  return;
 }
 p = d_path(path, pathname, PATH_MAX+11);
 if (IS_ERR(p)) {

  audit_log_string(ab, "<too_long>");
 } else
  audit_log_untrustedstring(ab, p);
 kfree(pathname);
}

void audit_log_session_info(struct audit_buffer *ab)
{
 unsigned int sessionid = audit_get_sessionid(current);
 uid_t auid = from_kuid(&init_user_ns, audit_get_loginuid(current));

 audit_log_format(ab, " auid=%u ses=%u", auid, sessionid);
}

void audit_log_key(struct audit_buffer *ab, char *key)
{
 audit_log_format(ab, " key=");
 if (key)
  audit_log_untrustedstring(ab, key);
 else
  audit_log_format(ab, "(null)");
}

void audit_log_cap(struct audit_buffer *ab, char *prefix, kernel_cap_t *cap)
{
 int i;

 audit_log_format(ab, " %s=", prefix);
 CAP_FOR_EACH_U32(i) {
  audit_log_format(ab, "%08x",
     cap->cap[CAP_LAST_U32 - i]);
 }
}

static void audit_log_fcaps(struct audit_buffer *ab, struct audit_names *name)
{
 kernel_cap_t *perm = &name->fcap.permitted;
 kernel_cap_t *inh = &name->fcap.inheritable;
 int log = 0;

 if (!cap_isclear(*perm)) {
  audit_log_cap(ab, "cap_fp", perm);
  log = 1;
 }
 if (!cap_isclear(*inh)) {
  audit_log_cap(ab, "cap_fi", inh);
  log = 1;
 }

 if (log)
  audit_log_format(ab, " cap_fe=%d cap_fver=%x",
     name->fcap.fE, name->fcap_ver);
}

static inline int audit_copy_fcaps(struct audit_names *name,
       const struct dentry *dentry)
{
 struct cpu_vfs_cap_data caps;
 int rc;

 if (!dentry)
  return 0;

 rc = get_vfs_caps_from_disk(dentry, &caps);
 if (rc)
  return rc;

 name->fcap.permitted = caps.permitted;
 name->fcap.inheritable = caps.inheritable;
 name->fcap.fE = !!(caps.magic_etc & VFS_CAP_FLAGS_EFFECTIVE);
 name->fcap_ver = (caps.magic_etc & VFS_CAP_REVISION_MASK) >>
    VFS_CAP_REVISION_SHIFT;

 return 0;
}


void audit_copy_inode(struct audit_names *name, const struct dentry *dentry,
        struct inode *inode)
{
 name->ino = inode->i_ino;
 name->dev = inode->i_sb->s_dev;
 name->mode = inode->i_mode;
 name->uid = inode->i_uid;
 name->gid = inode->i_gid;
 name->rdev = inode->i_rdev;
 security_inode_getsecid(inode, &name->osid);
 audit_copy_fcaps(name, dentry);
}
void audit_log_name(struct audit_context *context, struct audit_names *n,
      struct path *path, int record_num, int *call_panic)
{
 struct audit_buffer *ab;
 ab = audit_log_start(context, GFP_KERNEL, AUDIT_PATH);
 if (!ab)
  return;

 audit_log_format(ab, "item=%d", record_num);

 if (path)
  audit_log_d_path(ab, " name=", path);
 else if (n->name) {
  switch (n->name_len) {
  case AUDIT_NAME_FULL:

   audit_log_format(ab, " name=");
   audit_log_untrustedstring(ab, n->name->name);
   break;
  case 0:


   audit_log_d_path(ab, " name=", &context->pwd);
   break;
  default:

   audit_log_format(ab, " name=");
   audit_log_n_untrustedstring(ab, n->name->name,
          n->name_len);
  }
 } else
  audit_log_format(ab, " name=(null)");

 if (n->ino != AUDIT_INO_UNSET)
  audit_log_format(ab, " inode=%lu"
     " dev=%02x:%02x mode=%#ho"
     " ouid=%u ogid=%u rdev=%02x:%02x",
     n->ino,
     MAJOR(n->dev),
     MINOR(n->dev),
     n->mode,
     from_kuid(&init_user_ns, n->uid),
     from_kgid(&init_user_ns, n->gid),
     MAJOR(n->rdev),
     MINOR(n->rdev));
 if (n->osid != 0) {
  char *ctx = NULL;
  u32 len;
  if (security_secid_to_secctx(
   n->osid, &ctx, &len)) {
   audit_log_format(ab, " osid=%u", n->osid);
   if (call_panic)
    *call_panic = 2;
  } else {
   audit_log_format(ab, " obj=%s", ctx);
   security_release_secctx(ctx, len);
  }
 }


 audit_log_format(ab, " nametype=");
 switch(n->type) {
 case AUDIT_TYPE_NORMAL:
  audit_log_format(ab, "NORMAL");
  break;
 case AUDIT_TYPE_PARENT:
  audit_log_format(ab, "PARENT");
  break;
 case AUDIT_TYPE_CHILD_DELETE:
  audit_log_format(ab, "DELETE");
  break;
 case AUDIT_TYPE_CHILD_CREATE:
  audit_log_format(ab, "CREATE");
  break;
 default:
  audit_log_format(ab, "UNKNOWN");
  break;
 }

 audit_log_fcaps(ab, n);
 audit_log_end(ab);
}

int audit_log_task_context(struct audit_buffer *ab)
{
 char *ctx = NULL;
 unsigned len;
 int error;
 u32 sid;

 security_task_getsecid(current, &sid);
 if (!sid)
  return 0;

 error = security_secid_to_secctx(sid, &ctx, &len);
 if (error) {
  if (error != -EINVAL)
   goto error_path;
  return 0;
 }

 audit_log_format(ab, " subj=%s", ctx);
 security_release_secctx(ctx, len);
 return 0;

error_path:
 audit_panic("error in audit_log_task_context");
 return error;
}
EXPORT_SYMBOL(audit_log_task_context);

void audit_log_d_path_exe(struct audit_buffer *ab,
     struct mm_struct *mm)
{
 struct file *exe_file;

 if (!mm)
  goto out_null;

 exe_file = get_mm_exe_file(mm);
 if (!exe_file)
  goto out_null;

 audit_log_d_path(ab, " exe=", &exe_file->f_path);
 fput(exe_file);
 return;
out_null:
 audit_log_format(ab, " exe=(null)");
}

struct tty_struct *audit_get_tty(struct task_struct *tsk)
{
 struct tty_struct *tty = NULL;
 unsigned long flags;

 spin_lock_irqsave(&tsk->sighand->siglock, flags);
 if (tsk->signal)
  tty = tty_kref_get(tsk->signal->tty);
 spin_unlock_irqrestore(&tsk->sighand->siglock, flags);
 return tty;
}

void audit_put_tty(struct tty_struct *tty)
{
 tty_kref_put(tty);
}

void audit_log_task_info(struct audit_buffer *ab, struct task_struct *tsk)
{
 const struct cred *cred;
 char comm[sizeof(tsk->comm)];
 struct tty_struct *tty;

 if (!ab)
  return;


 cred = current_cred();
 tty = audit_get_tty(tsk);
 audit_log_format(ab,
    " ppid=%d pid=%d auid=%u uid=%u gid=%u"
    " euid=%u suid=%u fsuid=%u"
    " egid=%u sgid=%u fsgid=%u tty=%s ses=%u",
    task_ppid_nr(tsk),
    task_pid_nr(tsk),
    from_kuid(&init_user_ns, audit_get_loginuid(tsk)),
    from_kuid(&init_user_ns, cred->uid),
    from_kgid(&init_user_ns, cred->gid),
    from_kuid(&init_user_ns, cred->euid),
    from_kuid(&init_user_ns, cred->suid),
    from_kuid(&init_user_ns, cred->fsuid),
    from_kgid(&init_user_ns, cred->egid),
    from_kgid(&init_user_ns, cred->sgid),
    from_kgid(&init_user_ns, cred->fsgid),
    tty ? tty_name(tty) : "(none)",
    audit_get_sessionid(tsk));
 audit_put_tty(tty);
 audit_log_format(ab, " comm=");
 audit_log_untrustedstring(ab, get_task_comm(comm, tsk));
 audit_log_d_path_exe(ab, tsk->mm);
 audit_log_task_context(ab);
}
EXPORT_SYMBOL(audit_log_task_info);






void audit_log_link_denied(const char *operation, struct path *link)
{
 struct audit_buffer *ab;
 struct audit_names *name;

 name = kzalloc(sizeof(*name), GFP_NOFS);
 if (!name)
  return;


 ab = audit_log_start(current->audit_context, GFP_KERNEL,
        AUDIT_ANOM_LINK);
 if (!ab)
  goto out;
 audit_log_format(ab, "op=%s", operation);
 audit_log_task_info(ab, current);
 audit_log_format(ab, " res=0");
 audit_log_end(ab);


 name->type = AUDIT_TYPE_NORMAL;
 audit_copy_inode(name, link->dentry, d_backing_inode(link->dentry));
 audit_log_name(current->audit_context, name, link, 0, NULL);
out:
 kfree(name);
}
void audit_log_end(struct audit_buffer *ab)
{
 if (!ab)
  return;
 if (!audit_rate_check()) {
  audit_log_lost("rate limit exceeded");
 } else {
  struct nlmsghdr *nlh = nlmsg_hdr(ab->skb);

  nlh->nlmsg_len = ab->skb->len;
  kauditd_send_multicast_skb(ab->skb, ab->gfp_mask);
  nlh->nlmsg_len -= NLMSG_HDRLEN;

  if (audit_pid) {
   skb_queue_tail(&audit_skb_queue, ab->skb);
   wake_up_interruptible(&kauditd_wait);
  } else {
   audit_printk_skb(ab->skb);
  }
  ab->skb = NULL;
 }
 audit_buffer_free(ab);
}
void audit_log(struct audit_context *ctx, gfp_t gfp_mask, int type,
        const char *fmt, ...)
{
 struct audit_buffer *ab;
 va_list args;

 ab = audit_log_start(ctx, gfp_mask, type);
 if (ab) {
  va_start(args, fmt);
  audit_log_vformat(ab, fmt, args);
  va_end(args);
  audit_log_end(ab);
 }
}

void audit_log_secctx(struct audit_buffer *ab, u32 secid)
{
 u32 len;
 char *secctx;

 if (security_secid_to_secctx(secid, &secctx, &len)) {
  audit_panic("Cannot convert secid to context");
 } else {
  audit_log_format(ab, " obj=%s", secctx);
  security_release_secctx(secctx, len);
 }
}
EXPORT_SYMBOL(audit_log_secctx);

EXPORT_SYMBOL(audit_log_start);
EXPORT_SYMBOL(audit_log_end);
EXPORT_SYMBOL(audit_log_format);
EXPORT_SYMBOL(audit_log);

struct list_head audit_filter_list[AUDIT_NR_FILTERS] = {
 LIST_HEAD_INIT(audit_filter_list[0]),
 LIST_HEAD_INIT(audit_filter_list[1]),
 LIST_HEAD_INIT(audit_filter_list[2]),
 LIST_HEAD_INIT(audit_filter_list[3]),
 LIST_HEAD_INIT(audit_filter_list[4]),
 LIST_HEAD_INIT(audit_filter_list[5]),
};
static struct list_head audit_rules_list[AUDIT_NR_FILTERS] = {
 LIST_HEAD_INIT(audit_rules_list[0]),
 LIST_HEAD_INIT(audit_rules_list[1]),
 LIST_HEAD_INIT(audit_rules_list[2]),
 LIST_HEAD_INIT(audit_rules_list[3]),
 LIST_HEAD_INIT(audit_rules_list[4]),
 LIST_HEAD_INIT(audit_rules_list[5]),
};

DEFINE_MUTEX(audit_filter_mutex);

static void audit_free_lsm_field(struct audit_field *f)
{
 switch (f->type) {
 case AUDIT_SUBJ_USER:
 case AUDIT_SUBJ_ROLE:
 case AUDIT_SUBJ_TYPE:
 case AUDIT_SUBJ_SEN:
 case AUDIT_SUBJ_CLR:
 case AUDIT_OBJ_USER:
 case AUDIT_OBJ_ROLE:
 case AUDIT_OBJ_TYPE:
 case AUDIT_OBJ_LEV_LOW:
 case AUDIT_OBJ_LEV_HIGH:
  kfree(f->lsm_str);
  security_audit_rule_free(f->lsm_rule);
 }
}

static inline void audit_free_rule(struct audit_entry *e)
{
 int i;
 struct audit_krule *erule = &e->rule;


 if (erule->watch)
  audit_put_watch(erule->watch);
 if (erule->fields)
  for (i = 0; i < erule->field_count; i++)
   audit_free_lsm_field(&erule->fields[i]);
 kfree(erule->fields);
 kfree(erule->filterkey);
 kfree(e);
}

void audit_free_rule_rcu(struct rcu_head *head)
{
 struct audit_entry *e = container_of(head, struct audit_entry, rcu);
 audit_free_rule(e);
}


static inline struct audit_entry *audit_init_entry(u32 field_count)
{
 struct audit_entry *entry;
 struct audit_field *fields;

 entry = kzalloc(sizeof(*entry), GFP_KERNEL);
 if (unlikely(!entry))
  return NULL;

 fields = kcalloc(field_count, sizeof(*fields), GFP_KERNEL);
 if (unlikely(!fields)) {
  kfree(entry);
  return NULL;
 }
 entry->rule.fields = fields;

 return entry;
}



char *audit_unpack_string(void **bufp, size_t *remain, size_t len)
{
 char *str;

 if (!*bufp || (len == 0) || (len > *remain))
  return ERR_PTR(-EINVAL);




 if (len > PATH_MAX)
  return ERR_PTR(-ENAMETOOLONG);

 str = kmalloc(len + 1, GFP_KERNEL);
 if (unlikely(!str))
  return ERR_PTR(-ENOMEM);

 memcpy(str, *bufp, len);
 str[len] = 0;
 *bufp += len;
 *remain -= len;

 return str;
}


static inline int audit_to_inode(struct audit_krule *krule,
     struct audit_field *f)
{
 if (krule->listnr != AUDIT_FILTER_EXIT ||
     krule->inode_f || krule->watch || krule->tree ||
     (f->op != Audit_equal && f->op != Audit_not_equal))
  return -EINVAL;

 krule->inode_f = f;
 return 0;
}

static __u32 *classes[AUDIT_SYSCALL_CLASSES];

int __init audit_register_class(int class, unsigned *list)
{
 __u32 *p = kcalloc(AUDIT_BITMASK_SIZE, sizeof(__u32), GFP_KERNEL);
 if (!p)
  return -ENOMEM;
 while (*list != ~0U) {
  unsigned n = *list++;
  if (n >= AUDIT_BITMASK_SIZE * 32 - AUDIT_SYSCALL_CLASSES) {
   kfree(p);
   return -EINVAL;
  }
  p[AUDIT_WORD(n)] |= AUDIT_BIT(n);
 }
 if (class >= AUDIT_SYSCALL_CLASSES || classes[class]) {
  kfree(p);
  return -EINVAL;
 }
 classes[class] = p;
 return 0;
}

int audit_match_class(int class, unsigned syscall)
{
 if (unlikely(syscall >= AUDIT_BITMASK_SIZE * 32))
  return 0;
 if (unlikely(class >= AUDIT_SYSCALL_CLASSES || !classes[class]))
  return 0;
 return classes[class][AUDIT_WORD(syscall)] & AUDIT_BIT(syscall);
}

static inline int audit_match_class_bits(int class, u32 *mask)
{
 int i;

 if (classes[class]) {
  for (i = 0; i < AUDIT_BITMASK_SIZE; i++)
   if (mask[i] & classes[class][i])
    return 0;
 }
 return 1;
}

static int audit_match_signal(struct audit_entry *entry)
{
 struct audit_field *arch = entry->rule.arch_f;

 if (!arch) {


  return (audit_match_class_bits(AUDIT_CLASS_SIGNAL,
            entry->rule.mask) &&
   audit_match_class_bits(AUDIT_CLASS_SIGNAL_32,
            entry->rule.mask));
 }

 switch(audit_classify_arch(arch->val)) {
 case 0:
  return (audit_match_class_bits(AUDIT_CLASS_SIGNAL,
            entry->rule.mask));
 case 1:
  return (audit_match_class_bits(AUDIT_CLASS_SIGNAL_32,
            entry->rule.mask));
 default:
  return 1;
 }
}


static inline struct audit_entry *audit_to_entry_common(struct audit_rule_data *rule)
{
 unsigned listnr;
 struct audit_entry *entry;
 int i, err;

 err = -EINVAL;
 listnr = rule->flags & ~AUDIT_FILTER_PREPEND;
 switch(listnr) {
 default:
  goto exit_err;
 case AUDIT_FILTER_ENTRY:
  if (rule->action == AUDIT_ALWAYS)
   goto exit_err;
 case AUDIT_FILTER_EXIT:
 case AUDIT_FILTER_TASK:
 case AUDIT_FILTER_USER:
 case AUDIT_FILTER_TYPE:
  ;
 }
 if (unlikely(rule->action == AUDIT_POSSIBLE)) {
  pr_err("AUDIT_POSSIBLE is deprecated\n");
  goto exit_err;
 }
 if (rule->action != AUDIT_NEVER && rule->action != AUDIT_ALWAYS)
  goto exit_err;
 if (rule->field_count > AUDIT_MAX_FIELDS)
  goto exit_err;

 err = -ENOMEM;
 entry = audit_init_entry(rule->field_count);
 if (!entry)
  goto exit_err;

 entry->rule.flags = rule->flags & AUDIT_FILTER_PREPEND;
 entry->rule.listnr = listnr;
 entry->rule.action = rule->action;
 entry->rule.field_count = rule->field_count;

 for (i = 0; i < AUDIT_BITMASK_SIZE; i++)
  entry->rule.mask[i] = rule->mask[i];

 for (i = 0; i < AUDIT_SYSCALL_CLASSES; i++) {
  int bit = AUDIT_BITMASK_SIZE * 32 - i - 1;
  __u32 *p = &entry->rule.mask[AUDIT_WORD(bit)];
  __u32 *class;

  if (!(*p & AUDIT_BIT(bit)))
   continue;
  *p &= ~AUDIT_BIT(bit);
  class = classes[i];
  if (class) {
   int j;
   for (j = 0; j < AUDIT_BITMASK_SIZE; j++)
    entry->rule.mask[j] |= class[j];
  }
 }

 return entry;

exit_err:
 return ERR_PTR(err);
}

static u32 audit_ops[] =
{
 [Audit_equal] = AUDIT_EQUAL,
 [Audit_not_equal] = AUDIT_NOT_EQUAL,
 [Audit_bitmask] = AUDIT_BIT_MASK,
 [Audit_bittest] = AUDIT_BIT_TEST,
 [Audit_lt] = AUDIT_LESS_THAN,
 [Audit_gt] = AUDIT_GREATER_THAN,
 [Audit_le] = AUDIT_LESS_THAN_OR_EQUAL,
 [Audit_ge] = AUDIT_GREATER_THAN_OR_EQUAL,
};

static u32 audit_to_op(u32 op)
{
 u32 n;
 for (n = Audit_equal; n < Audit_bad && audit_ops[n] != op; n++)
  ;
 return n;
}


static int audit_field_valid(struct audit_entry *entry, struct audit_field *f)
{
 switch(f->type) {
 case AUDIT_MSGTYPE:
  if (entry->rule.listnr != AUDIT_FILTER_TYPE &&
      entry->rule.listnr != AUDIT_FILTER_USER)
   return -EINVAL;
  break;
 };

 switch(f->type) {
 default:
  return -EINVAL;
 case AUDIT_UID:
 case AUDIT_EUID:
 case AUDIT_SUID:
 case AUDIT_FSUID:
 case AUDIT_LOGINUID:
 case AUDIT_OBJ_UID:
 case AUDIT_GID:
 case AUDIT_EGID:
 case AUDIT_SGID:
 case AUDIT_FSGID:
 case AUDIT_OBJ_GID:
 case AUDIT_PID:
 case AUDIT_PERS:
 case AUDIT_MSGTYPE:
 case AUDIT_PPID:
 case AUDIT_DEVMAJOR:
 case AUDIT_DEVMINOR:
 case AUDIT_EXIT:
 case AUDIT_SUCCESS:
 case AUDIT_INODE:

  if (f->op == Audit_bitmask || f->op == Audit_bittest)
   return -EINVAL;
  break;
 case AUDIT_ARG0:
 case AUDIT_ARG1:
 case AUDIT_ARG2:
 case AUDIT_ARG3:
 case AUDIT_SUBJ_USER:
 case AUDIT_SUBJ_ROLE:
 case AUDIT_SUBJ_TYPE:
 case AUDIT_SUBJ_SEN:
 case AUDIT_SUBJ_CLR:
 case AUDIT_OBJ_USER:
 case AUDIT_OBJ_ROLE:
 case AUDIT_OBJ_TYPE:
 case AUDIT_OBJ_LEV_LOW:
 case AUDIT_OBJ_LEV_HIGH:
 case AUDIT_WATCH:
 case AUDIT_DIR:
 case AUDIT_FILTERKEY:
  break;
 case AUDIT_LOGINUID_SET:
  if ((f->val != 0) && (f->val != 1))
   return -EINVAL;

 case AUDIT_ARCH:
  if (f->op != Audit_not_equal && f->op != Audit_equal)
   return -EINVAL;
  break;
 case AUDIT_PERM:
  if (f->val & ~15)
   return -EINVAL;
  break;
 case AUDIT_FILETYPE:
  if (f->val & ~S_IFMT)
   return -EINVAL;
  break;
 case AUDIT_FIELD_COMPARE:
  if (f->val > AUDIT_MAX_FIELD_COMPARE)
   return -EINVAL;
  break;
 case AUDIT_EXE:
  if (f->op != Audit_equal)
   return -EINVAL;
  if (entry->rule.listnr != AUDIT_FILTER_EXIT)
   return -EINVAL;
  break;
 };
 return 0;
}


static struct audit_entry *audit_data_to_entry(struct audit_rule_data *data,
            size_t datasz)
{
 int err = 0;
 struct audit_entry *entry;
 void *bufp;
 size_t remain = datasz - sizeof(struct audit_rule_data);
 int i;
 char *str;
 struct audit_fsnotify_mark *audit_mark;

 entry = audit_to_entry_common(data);
 if (IS_ERR(entry))
  goto exit_nofree;

 bufp = data->buf;
 for (i = 0; i < data->field_count; i++) {
  struct audit_field *f = &entry->rule.fields[i];

  err = -EINVAL;

  f->op = audit_to_op(data->fieldflags[i]);
  if (f->op == Audit_bad)
   goto exit_free;

  f->type = data->fields[i];
  f->val = data->values[i];


  if ((f->type == AUDIT_LOGINUID) && (f->val == AUDIT_UID_UNSET)) {
   f->type = AUDIT_LOGINUID_SET;
   f->val = 0;
   entry->rule.pflags |= AUDIT_LOGINUID_LEGACY;
  }

  err = audit_field_valid(entry, f);
  if (err)
   goto exit_free;

  err = -EINVAL;
  switch (f->type) {
  case AUDIT_LOGINUID:
  case AUDIT_UID:
  case AUDIT_EUID:
  case AUDIT_SUID:
  case AUDIT_FSUID:
  case AUDIT_OBJ_UID:
   f->uid = make_kuid(current_user_ns(), f->val);
   if (!uid_valid(f->uid))
    goto exit_free;
   break;
  case AUDIT_GID:
  case AUDIT_EGID:
  case AUDIT_SGID:
  case AUDIT_FSGID:
  case AUDIT_OBJ_GID:
   f->gid = make_kgid(current_user_ns(), f->val);
   if (!gid_valid(f->gid))
    goto exit_free;
   break;
  case AUDIT_ARCH:
   entry->rule.arch_f = f;
   break;
  case AUDIT_SUBJ_USER:
  case AUDIT_SUBJ_ROLE:
  case AUDIT_SUBJ_TYPE:
  case AUDIT_SUBJ_SEN:
  case AUDIT_SUBJ_CLR:
  case AUDIT_OBJ_USER:
  case AUDIT_OBJ_ROLE:
  case AUDIT_OBJ_TYPE:
  case AUDIT_OBJ_LEV_LOW:
  case AUDIT_OBJ_LEV_HIGH:
   str = audit_unpack_string(&bufp, &remain, f->val);
   if (IS_ERR(str))
    goto exit_free;
   entry->rule.buflen += f->val;

   err = security_audit_rule_init(f->type, f->op, str,
             (void **)&f->lsm_rule);


   if (err == -EINVAL) {
    pr_warn("audit rule for LSM \'%s\' is invalid\n",
     str);
    err = 0;
   }
   if (err) {
    kfree(str);
    goto exit_free;
   } else
    f->lsm_str = str;
   break;
  case AUDIT_WATCH:
   str = audit_unpack_string(&bufp, &remain, f->val);
   if (IS_ERR(str))
    goto exit_free;
   entry->rule.buflen += f->val;

   err = audit_to_watch(&entry->rule, str, f->val, f->op);
   if (err) {
    kfree(str);
    goto exit_free;
   }
   break;
  case AUDIT_DIR:
   str = audit_unpack_string(&bufp, &remain, f->val);
   if (IS_ERR(str))
    goto exit_free;
   entry->rule.buflen += f->val;

   err = audit_make_tree(&entry->rule, str, f->op);
   kfree(str);
   if (err)
    goto exit_free;
   break;
  case AUDIT_INODE:
   err = audit_to_inode(&entry->rule, f);
   if (err)
    goto exit_free;
   break;
  case AUDIT_FILTERKEY:
   if (entry->rule.filterkey || f->val > AUDIT_MAX_KEY_LEN)
    goto exit_free;
   str = audit_unpack_string(&bufp, &remain, f->val);
   if (IS_ERR(str))
    goto exit_free;
   entry->rule.buflen += f->val;
   entry->rule.filterkey = str;
   break;
  case AUDIT_EXE:
   if (entry->rule.exe || f->val > PATH_MAX)
    goto exit_free;
   str = audit_unpack_string(&bufp, &remain, f->val);
   if (IS_ERR(str)) {
    err = PTR_ERR(str);
    goto exit_free;
   }
   entry->rule.buflen += f->val;

   audit_mark = audit_alloc_mark(&entry->rule, str, f->val);
   if (IS_ERR(audit_mark)) {
    kfree(str);
    err = PTR_ERR(audit_mark);
    goto exit_free;
   }
   entry->rule.exe = audit_mark;
   break;
  }
 }

 if (entry->rule.inode_f && entry->rule.inode_f->op == Audit_not_equal)
  entry->rule.inode_f = NULL;

exit_nofree:
 return entry;

exit_free:
 if (entry->rule.tree)
  audit_put_tree(entry->rule.tree);
 if (entry->rule.exe)
  audit_remove_mark(entry->rule.exe);
 audit_free_rule(entry);
 return ERR_PTR(err);
}


static inline size_t audit_pack_string(void **bufp, const char *str)
{
 size_t len = strlen(str);

 memcpy(*bufp, str, len);
 *bufp += len;

 return len;
}


static struct audit_rule_data *audit_krule_to_data(struct audit_krule *krule)
{
 struct audit_rule_data *data;
 void *bufp;
 int i;

 data = kmalloc(sizeof(*data) + krule->buflen, GFP_KERNEL);
 if (unlikely(!data))
  return NULL;
 memset(data, 0, sizeof(*data));

 data->flags = krule->flags | krule->listnr;
 data->action = krule->action;
 data->field_count = krule->field_count;
 bufp = data->buf;
 for (i = 0; i < data->field_count; i++) {
  struct audit_field *f = &krule->fields[i];

  data->fields[i] = f->type;
  data->fieldflags[i] = audit_ops[f->op];
  switch(f->type) {
  case AUDIT_SUBJ_USER:
  case AUDIT_SUBJ_ROLE:
  case AUDIT_SUBJ_TYPE:
  case AUDIT_SUBJ_SEN:
  case AUDIT_SUBJ_CLR:
  case AUDIT_OBJ_USER:
  case AUDIT_OBJ_ROLE:
  case AUDIT_OBJ_TYPE:
  case AUDIT_OBJ_LEV_LOW:
  case AUDIT_OBJ_LEV_HIGH:
   data->buflen += data->values[i] =
    audit_pack_string(&bufp, f->lsm_str);
   break;
  case AUDIT_WATCH:
   data->buflen += data->values[i] =
    audit_pack_string(&bufp,
        audit_watch_path(krule->watch));
   break;
  case AUDIT_DIR:
   data->buflen += data->values[i] =
    audit_pack_string(&bufp,
        audit_tree_path(krule->tree));
   break;
  case AUDIT_FILTERKEY:
   data->buflen += data->values[i] =
    audit_pack_string(&bufp, krule->filterkey);
   break;
  case AUDIT_EXE:
   data->buflen += data->values[i] =
    audit_pack_string(&bufp, audit_mark_path(krule->exe));
   break;
  case AUDIT_LOGINUID_SET:
   if (krule->pflags & AUDIT_LOGINUID_LEGACY && !f->val) {
    data->fields[i] = AUDIT_LOGINUID;
    data->values[i] = AUDIT_UID_UNSET;
    break;
   }

  default:
   data->values[i] = f->val;
  }
 }
 for (i = 0; i < AUDIT_BITMASK_SIZE; i++) data->mask[i] = krule->mask[i];

 return data;
}



static int audit_compare_rule(struct audit_krule *a, struct audit_krule *b)
{
 int i;

 if (a->flags != b->flags ||
     a->pflags != b->pflags ||
     a->listnr != b->listnr ||
     a->action != b->action ||
     a->field_count != b->field_count)
  return 1;

 for (i = 0; i < a->field_count; i++) {
  if (a->fields[i].type != b->fields[i].type ||
      a->fields[i].op != b->fields[i].op)
   return 1;

  switch(a->fields[i].type) {
  case AUDIT_SUBJ_USER:
  case AUDIT_SUBJ_ROLE:
  case AUDIT_SUBJ_TYPE:
  case AUDIT_SUBJ_SEN:
  case AUDIT_SUBJ_CLR:
  case AUDIT_OBJ_USER:
  case AUDIT_OBJ_ROLE:
  case AUDIT_OBJ_TYPE:
  case AUDIT_OBJ_LEV_LOW:
  case AUDIT_OBJ_LEV_HIGH:
   if (strcmp(a->fields[i].lsm_str, b->fields[i].lsm_str))
    return 1;
   break;
  case AUDIT_WATCH:
   if (strcmp(audit_watch_path(a->watch),
       audit_watch_path(b->watch)))
    return 1;
   break;
  case AUDIT_DIR:
   if (strcmp(audit_tree_path(a->tree),
       audit_tree_path(b->tree)))
    return 1;
   break;
  case AUDIT_FILTERKEY:

   if (strcmp(a->filterkey, b->filterkey))
    return 1;
   break;
  case AUDIT_EXE:

   if (strcmp(audit_mark_path(a->exe),
       audit_mark_path(b->exe)))
    return 1;
   break;
  case AUDIT_UID:
  case AUDIT_EUID:
  case AUDIT_SUID:
  case AUDIT_FSUID:
  case AUDIT_LOGINUID:
  case AUDIT_OBJ_UID:
   if (!uid_eq(a->fields[i].uid, b->fields[i].uid))
    return 1;
   break;
  case AUDIT_GID:
  case AUDIT_EGID:
  case AUDIT_SGID:
  case AUDIT_FSGID:
  case AUDIT_OBJ_GID:
   if (!gid_eq(a->fields[i].gid, b->fields[i].gid))
    return 1;
   break;
  default:
   if (a->fields[i].val != b->fields[i].val)
    return 1;
  }
 }

 for (i = 0; i < AUDIT_BITMASK_SIZE; i++)
  if (a->mask[i] != b->mask[i])
   return 1;

 return 0;
}



static inline int audit_dupe_lsm_field(struct audit_field *df,
        struct audit_field *sf)
{
 int ret = 0;
 char *lsm_str;


 lsm_str = kstrdup(sf->lsm_str, GFP_KERNEL);
 if (unlikely(!lsm_str))
  return -ENOMEM;
 df->lsm_str = lsm_str;


 ret = security_audit_rule_init(df->type, df->op, df->lsm_str,
           (void **)&df->lsm_rule);


 if (ret == -EINVAL) {
  pr_warn("audit rule for LSM \'%s\' is invalid\n",
   df->lsm_str);
  ret = 0;
 }

 return ret;
}







struct audit_entry *audit_dupe_rule(struct audit_krule *old)
{
 u32 fcount = old->field_count;
 struct audit_entry *entry;
 struct audit_krule *new;
 char *fk;
 int i, err = 0;

 entry = audit_init_entry(fcount);
 if (unlikely(!entry))
  return ERR_PTR(-ENOMEM);

 new = &entry->rule;
 new->flags = old->flags;
 new->pflags = old->pflags;
 new->listnr = old->listnr;
 new->action = old->action;
 for (i = 0; i < AUDIT_BITMASK_SIZE; i++)
  new->mask[i] = old->mask[i];
 new->prio = old->prio;
 new->buflen = old->buflen;
 new->inode_f = old->inode_f;
 new->field_count = old->field_count;
 new->tree = old->tree;
 memcpy(new->fields, old->fields, sizeof(struct audit_field) * fcount);



 for (i = 0; i < fcount; i++) {
  switch (new->fields[i].type) {
  case AUDIT_SUBJ_USER:
  case AUDIT_SUBJ_ROLE:
  case AUDIT_SUBJ_TYPE:
  case AUDIT_SUBJ_SEN:
  case AUDIT_SUBJ_CLR:
  case AUDIT_OBJ_USER:
  case AUDIT_OBJ_ROLE:
  case AUDIT_OBJ_TYPE:
  case AUDIT_OBJ_LEV_LOW:
  case AUDIT_OBJ_LEV_HIGH:
   err = audit_dupe_lsm_field(&new->fields[i],
             &old->fields[i]);
   break;
  case AUDIT_FILTERKEY:
   fk = kstrdup(old->filterkey, GFP_KERNEL);
   if (unlikely(!fk))
    err = -ENOMEM;
   else
    new->filterkey = fk;
   break;
  case AUDIT_EXE:
   err = audit_dupe_exe(new, old);
   break;
  }
  if (err) {
   if (new->exe)
    audit_remove_mark(new->exe);
   audit_free_rule(entry);
   return ERR_PTR(err);
  }
 }

 if (old->watch) {
  audit_get_watch(old->watch);
  new->watch = old->watch;
 }

 return entry;
}



static struct audit_entry *audit_find_rule(struct audit_entry *entry,
        struct list_head **p)
{
 struct audit_entry *e, *found = NULL;
 struct list_head *list;
 int h;

 if (entry->rule.inode_f) {
  h = audit_hash_ino(entry->rule.inode_f->val);
  *p = list = &audit_inode_hash[h];
 } else if (entry->rule.watch) {

  for (h = 0; h < AUDIT_INODE_BUCKETS; h++) {
   list = &audit_inode_hash[h];
   list_for_each_entry(e, list, list)
    if (!audit_compare_rule(&entry->rule, &e->rule)) {
     found = e;
     goto out;
    }
  }
  goto out;
 } else {
  *p = list = &audit_filter_list[entry->rule.listnr];
 }

 list_for_each_entry(e, list, list)
  if (!audit_compare_rule(&entry->rule, &e->rule)) {
   found = e;
   goto out;
  }

out:
 return found;
}

static u64 prio_low = ~0ULL/2;
static u64 prio_high = ~0ULL/2 - 1;


static inline int audit_add_rule(struct audit_entry *entry)
{
 struct audit_entry *e;
 struct audit_watch *watch = entry->rule.watch;
 struct audit_tree *tree = entry->rule.tree;
 struct list_head *list;
 int err = 0;
 int dont_count = 0;


 if (entry->rule.listnr == AUDIT_FILTER_USER ||
  entry->rule.listnr == AUDIT_FILTER_TYPE)
  dont_count = 1;

 mutex_lock(&audit_filter_mutex);
 e = audit_find_rule(entry, &list);
 if (e) {
  mutex_unlock(&audit_filter_mutex);
  err = -EEXIST;

  if (tree)
   audit_put_tree(tree);
  return err;
 }

 if (watch) {

  err = audit_add_watch(&entry->rule, &list);
  if (err) {
   mutex_unlock(&audit_filter_mutex);




   if (tree)
    audit_put_tree(tree);
   return err;
  }
 }
 if (tree) {
  err = audit_add_tree_rule(&entry->rule);
  if (err) {
   mutex_unlock(&audit_filter_mutex);
   return err;
  }
 }

 entry->rule.prio = ~0ULL;
 if (entry->rule.listnr == AUDIT_FILTER_EXIT) {
  if (entry->rule.flags & AUDIT_FILTER_PREPEND)
   entry->rule.prio = ++prio_high;
  else
   entry->rule.prio = --prio_low;
 }

 if (entry->rule.flags & AUDIT_FILTER_PREPEND) {
  list_add(&entry->rule.list,
    &audit_rules_list[entry->rule.listnr]);
  list_add_rcu(&entry->list, list);
  entry->rule.flags &= ~AUDIT_FILTER_PREPEND;
 } else {
  list_add_tail(&entry->rule.list,
         &audit_rules_list[entry->rule.listnr]);
  list_add_tail_rcu(&entry->list, list);
 }
 if (!dont_count)
  audit_n_rules++;

 if (!audit_match_signal(entry))
  audit_signals++;
 mutex_unlock(&audit_filter_mutex);

 return err;
}


int audit_del_rule(struct audit_entry *entry)
{
 struct audit_entry *e;
 struct audit_tree *tree = entry->rule.tree;
 struct list_head *list;
 int ret = 0;
 int dont_count = 0;


 if (entry->rule.listnr == AUDIT_FILTER_USER ||
  entry->rule.listnr == AUDIT_FILTER_TYPE)
  dont_count = 1;

 mutex_lock(&audit_filter_mutex);
 e = audit_find_rule(entry, &list);
 if (!e) {
  ret = -ENOENT;
  goto out;
 }

 if (e->rule.watch)
  audit_remove_watch_rule(&e->rule);

 if (e->rule.tree)
  audit_remove_tree_rule(&e->rule);

 if (e->rule.exe)
  audit_remove_mark_rule(&e->rule);

 if (!dont_count)
  audit_n_rules--;

 if (!audit_match_signal(entry))
  audit_signals--;

 list_del_rcu(&e->list);
 list_del(&e->rule.list);
 call_rcu(&e->rcu, audit_free_rule_rcu);

out:
 mutex_unlock(&audit_filter_mutex);

 if (tree)
  audit_put_tree(tree);

 return ret;
}


static void audit_list_rules(__u32 portid, int seq, struct sk_buff_head *q)
{
 struct sk_buff *skb;
 struct audit_krule *r;
 int i;



 for (i=0; i<AUDIT_NR_FILTERS; i++) {
  list_for_each_entry(r, &audit_rules_list[i], list) {
   struct audit_rule_data *data;

   data = audit_krule_to_data(r);
   if (unlikely(!data))
    break;
   skb = audit_make_reply(portid, seq, AUDIT_LIST_RULES,
            0, 1, data,
            sizeof(*data) + data->buflen);
   if (skb)
    skb_queue_tail(q, skb);
   kfree(data);
  }
 }
 skb = audit_make_reply(portid, seq, AUDIT_LIST_RULES, 1, 1, NULL, 0);
 if (skb)
  skb_queue_tail(q, skb);
}


static void audit_log_rule_change(char *action, struct audit_krule *rule, int res)
{
 struct audit_buffer *ab;
 uid_t loginuid = from_kuid(&init_user_ns, audit_get_loginuid(current));
 unsigned int sessionid = audit_get_sessionid(current);

 if (!audit_enabled)
  return;

 ab = audit_log_start(NULL, GFP_KERNEL, AUDIT_CONFIG_CHANGE);
 if (!ab)
  return;
 audit_log_format(ab, "auid=%u ses=%u" ,loginuid, sessionid);
 audit_log_task_context(ab);
 audit_log_format(ab, " op=");
 audit_log_string(ab, action);
 audit_log_key(ab, rule->filterkey);
 audit_log_format(ab, " list=%d res=%d", rule->listnr, res);
 audit_log_end(ab);
}
int audit_rule_change(int type, __u32 portid, int seq, void *data,
   size_t datasz)
{
 int err = 0;
 struct audit_entry *entry;

 entry = audit_data_to_entry(data, datasz);
 if (IS_ERR(entry))
  return PTR_ERR(entry);

 switch (type) {
 case AUDIT_ADD_RULE:
  err = audit_add_rule(entry);
  audit_log_rule_change("add_rule", &entry->rule, !err);
  break;
 case AUDIT_DEL_RULE:
  err = audit_del_rule(entry);
  audit_log_rule_change("remove_rule", &entry->rule, !err);
  break;
 default:
  err = -EINVAL;
  WARN_ON(1);
 }

 if (err || type == AUDIT_DEL_RULE) {
  if (entry->rule.exe)
   audit_remove_mark(entry->rule.exe);
  audit_free_rule(entry);
 }

 return err;
}






int audit_list_rules_send(struct sk_buff *request_skb, int seq)
{
 u32 portid = NETLINK_CB(request_skb).portid;
 struct net *net = sock_net(NETLINK_CB(request_skb).sk);
 struct task_struct *tsk;
 struct audit_netlink_list *dest;
 int err = 0;







 dest = kmalloc(sizeof(struct audit_netlink_list), GFP_KERNEL);
 if (!dest)
  return -ENOMEM;
 dest->net = get_net(net);
 dest->portid = portid;
 skb_queue_head_init(&dest->q);

 mutex_lock(&audit_filter_mutex);
 audit_list_rules(portid, seq, &dest->q);
 mutex_unlock(&audit_filter_mutex);

 tsk = kthread_run(audit_send_list, dest, "audit_send_list");
 if (IS_ERR(tsk)) {
  skb_queue_purge(&dest->q);
  kfree(dest);
  err = PTR_ERR(tsk);
 }

 return err;
}

int audit_comparator(u32 left, u32 op, u32 right)
{
 switch (op) {
 case Audit_equal:
  return (left == right);
 case Audit_not_equal:
  return (left != right);
 case Audit_lt:
  return (left < right);
 case Audit_le:
  return (left <= right);
 case Audit_gt:
  return (left > right);
 case Audit_ge:
  return (left >= right);
 case Audit_bitmask:
  return (left & right);
 case Audit_bittest:
  return ((left & right) == right);
 default:
  BUG();
  return 0;
 }
}

int audit_uid_comparator(kuid_t left, u32 op, kuid_t right)
{
 switch (op) {
 case Audit_equal:
  return uid_eq(left, right);
 case Audit_not_equal:
  return !uid_eq(left, right);
 case Audit_lt:
  return uid_lt(left, right);
 case Audit_le:
  return uid_lte(left, right);
 case Audit_gt:
  return uid_gt(left, right);
 case Audit_ge:
  return uid_gte(left, right);
 case Audit_bitmask:
 case Audit_bittest:
 default:
  BUG();
  return 0;
 }
}

int audit_gid_comparator(kgid_t left, u32 op, kgid_t right)
{
 switch (op) {
 case Audit_equal:
  return gid_eq(left, right);
 case Audit_not_equal:
  return !gid_eq(left, right);
 case Audit_lt:
  return gid_lt(left, right);
 case Audit_le:
  return gid_lte(left, right);
 case Audit_gt:
  return gid_gt(left, right);
 case Audit_ge:
  return gid_gte(left, right);
 case Audit_bitmask:
 case Audit_bittest:
 default:
  BUG();
  return 0;
 }
}





int parent_len(const char *path)
{
 int plen;
 const char *p;

 plen = strlen(path);

 if (plen == 0)
  return plen;


 p = path + plen - 1;
 while ((*p == '/') && (p > path))
  p--;


 while ((*p != '/') && (p > path))
  p--;


 if (*p == '/')
  p++;

 return p - path;
}
int audit_compare_dname_path(const char *dname, const char *path, int parentlen)
{
 int dlen, pathlen;
 const char *p;

 dlen = strlen(dname);
 pathlen = strlen(path);
 if (pathlen < dlen)
  return 1;

 parentlen = parentlen == AUDIT_NAME_FULL ? parent_len(path) : parentlen;
 if (pathlen - parentlen != dlen)
  return 1;

 p = path + parentlen;

 return strncmp(p, dname, dlen);
}

static int audit_filter_user_rules(struct audit_krule *rule, int type,
       enum audit_state *state)
{
 int i;

 for (i = 0; i < rule->field_count; i++) {
  struct audit_field *f = &rule->fields[i];
  pid_t pid;
  int result = 0;
  u32 sid;

  switch (f->type) {
  case AUDIT_PID:
   pid = task_pid_nr(current);
   result = audit_comparator(pid, f->op, f->val);
   break;
  case AUDIT_UID:
   result = audit_uid_comparator(current_uid(), f->op, f->uid);
   break;
  case AUDIT_GID:
   result = audit_gid_comparator(current_gid(), f->op, f->gid);
   break;
  case AUDIT_LOGINUID:
   result = audit_uid_comparator(audit_get_loginuid(current),
        f->op, f->uid);
   break;
  case AUDIT_LOGINUID_SET:
   result = audit_comparator(audit_loginuid_set(current),
        f->op, f->val);
   break;
  case AUDIT_MSGTYPE:
   result = audit_comparator(type, f->op, f->val);
   break;
  case AUDIT_SUBJ_USER:
  case AUDIT_SUBJ_ROLE:
  case AUDIT_SUBJ_TYPE:
  case AUDIT_SUBJ_SEN:
  case AUDIT_SUBJ_CLR:
   if (f->lsm_rule) {
    security_task_getsecid(current, &sid);
    result = security_audit_rule_match(sid,
           f->type,
           f->op,
           f->lsm_rule,
           NULL);
   }
   break;
  }

  if (!result)
   return 0;
 }
 switch (rule->action) {
 case AUDIT_NEVER: *state = AUDIT_DISABLED; break;
 case AUDIT_ALWAYS: *state = AUDIT_RECORD_CONTEXT; break;
 }
 return 1;
}

int audit_filter_user(int type)
{
 enum audit_state state = AUDIT_DISABLED;
 struct audit_entry *e;
 int rc, ret;

 ret = 1;

 rcu_read_lock();
 list_for_each_entry_rcu(e, &audit_filter_list[AUDIT_FILTER_USER], list) {
  rc = audit_filter_user_rules(&e->rule, type, &state);
  if (rc) {
   if (rc > 0 && state == AUDIT_DISABLED)
    ret = 0;
   break;
  }
 }
 rcu_read_unlock();

 return ret;
}

int audit_filter_type(int type)
{
 struct audit_entry *e;
 int result = 0;

 rcu_read_lock();
 if (list_empty(&audit_filter_list[AUDIT_FILTER_TYPE]))
  goto unlock_and_return;

 list_for_each_entry_rcu(e, &audit_filter_list[AUDIT_FILTER_TYPE],
    list) {
  int i;
  for (i = 0; i < e->rule.field_count; i++) {
   struct audit_field *f = &e->rule.fields[i];
   if (f->type == AUDIT_MSGTYPE) {
    result = audit_comparator(type, f->op, f->val);
    if (!result)
     break;
   }
  }
  if (result)
   goto unlock_and_return;
 }
unlock_and_return:
 rcu_read_unlock();
 return result;
}

static int update_lsm_rule(struct audit_krule *r)
{
 struct audit_entry *entry = container_of(r, struct audit_entry, rule);
 struct audit_entry *nentry;
 int err = 0;

 if (!security_audit_rule_known(r))
  return 0;

 nentry = audit_dupe_rule(r);
 if (entry->rule.exe)
  audit_remove_mark(entry->rule.exe);
 if (IS_ERR(nentry)) {


  err = PTR_ERR(nentry);
  audit_panic("error updating LSM filters");
  if (r->watch)
   list_del(&r->rlist);
  list_del_rcu(&entry->list);
  list_del(&r->list);
 } else {
  if (r->watch || r->tree)
   list_replace_init(&r->rlist, &nentry->rule.rlist);
  list_replace_rcu(&entry->list, &nentry->list);
  list_replace(&r->list, &nentry->rule.list);
 }
 call_rcu(&entry->rcu, audit_free_rule_rcu);

 return err;
}






int audit_update_lsm_rules(void)
{
 struct audit_krule *r, *n;
 int i, err = 0;


 mutex_lock(&audit_filter_mutex);

 for (i = 0; i < AUDIT_NR_FILTERS; i++) {
  list_for_each_entry_safe(r, n, &audit_rules_list[i], list) {
   int res = update_lsm_rule(r);
   if (!err)
    err = res;
  }
 }
 mutex_unlock(&audit_filter_mutex);

 return err;
}





struct audit_fsnotify_mark {
 dev_t dev;
 unsigned long ino;
 char *path;
 struct fsnotify_mark mark;
 struct audit_krule *rule;
};


static struct fsnotify_group *audit_fsnotify_group;


    FS_MOVE_SELF | FS_EVENT_ON_CHILD)

static void audit_fsnotify_mark_free(struct audit_fsnotify_mark *audit_mark)
{
 kfree(audit_mark->path);
 kfree(audit_mark);
}

static void audit_fsnotify_free_mark(struct fsnotify_mark *mark)
{
 struct audit_fsnotify_mark *audit_mark;

 audit_mark = container_of(mark, struct audit_fsnotify_mark, mark);
 audit_fsnotify_mark_free(audit_mark);
}

char *audit_mark_path(struct audit_fsnotify_mark *mark)
{
 return mark->path;
}

int audit_mark_compare(struct audit_fsnotify_mark *mark, unsigned long ino, dev_t dev)
{
 if (mark->ino == AUDIT_INO_UNSET)
  return 0;
 return (mark->ino == ino) && (mark->dev == dev);
}

static void audit_update_mark(struct audit_fsnotify_mark *audit_mark,
        struct inode *inode)
{
 audit_mark->dev = inode ? inode->i_sb->s_dev : AUDIT_DEV_UNSET;
 audit_mark->ino = inode ? inode->i_ino : AUDIT_INO_UNSET;
}

struct audit_fsnotify_mark *audit_alloc_mark(struct audit_krule *krule, char *pathname, int len)
{
 struct audit_fsnotify_mark *audit_mark;
 struct path path;
 struct dentry *dentry;
 struct inode *inode;
 int ret;

 if (pathname[0] != '/' || pathname[len-1] == '/')
  return ERR_PTR(-EINVAL);

 dentry = kern_path_locked(pathname, &path);
 if (IS_ERR(dentry))
  return (void *)dentry;
 inode = path.dentry->d_inode;
 inode_unlock(inode);

 audit_mark = kzalloc(sizeof(*audit_mark), GFP_KERNEL);
 if (unlikely(!audit_mark)) {
  audit_mark = ERR_PTR(-ENOMEM);
  goto out;
 }

 fsnotify_init_mark(&audit_mark->mark, audit_fsnotify_free_mark);
 audit_mark->mark.mask = AUDIT_FS_EVENTS;
 audit_mark->path = pathname;
 audit_update_mark(audit_mark, dentry->d_inode);
 audit_mark->rule = krule;

 ret = fsnotify_add_mark(&audit_mark->mark, audit_fsnotify_group, inode, NULL, true);
 if (ret < 0) {
  audit_fsnotify_mark_free(audit_mark);
  audit_mark = ERR_PTR(ret);
 }
out:
 dput(dentry);
 path_put(&path);
 return audit_mark;
}

static void audit_mark_log_rule_change(struct audit_fsnotify_mark *audit_mark, char *op)
{
 struct audit_buffer *ab;
 struct audit_krule *rule = audit_mark->rule;

 if (!audit_enabled)
  return;
 ab = audit_log_start(NULL, GFP_NOFS, AUDIT_CONFIG_CHANGE);
 if (unlikely(!ab))
  return;
 audit_log_format(ab, "auid=%u ses=%u op=",
    from_kuid(&init_user_ns, audit_get_loginuid(current)),
    audit_get_sessionid(current));
 audit_log_string(ab, op);
 audit_log_format(ab, " path=");
 audit_log_untrustedstring(ab, audit_mark->path);
 audit_log_key(ab, rule->filterkey);
 audit_log_format(ab, " list=%d res=1", rule->listnr);
 audit_log_end(ab);
}

void audit_remove_mark(struct audit_fsnotify_mark *audit_mark)
{
 fsnotify_destroy_mark(&audit_mark->mark, audit_fsnotify_group);
 fsnotify_put_mark(&audit_mark->mark);
}

void audit_remove_mark_rule(struct audit_krule *krule)
{
 struct audit_fsnotify_mark *mark = krule->exe;

 audit_remove_mark(mark);
}

static void audit_autoremove_mark_rule(struct audit_fsnotify_mark *audit_mark)
{
 struct audit_krule *rule = audit_mark->rule;
 struct audit_entry *entry = container_of(rule, struct audit_entry, rule);

 audit_mark_log_rule_change(audit_mark, "autoremove_rule");
 audit_del_rule(entry);
}


static int audit_mark_handle_event(struct fsnotify_group *group,
        struct inode *to_tell,
        struct fsnotify_mark *inode_mark,
        struct fsnotify_mark *vfsmount_mark,
        u32 mask, void *data, int data_type,
        const unsigned char *dname, u32 cookie)
{
 struct audit_fsnotify_mark *audit_mark;
 struct inode *inode = NULL;

 audit_mark = container_of(inode_mark, struct audit_fsnotify_mark, mark);

 BUG_ON(group != audit_fsnotify_group);

 switch (data_type) {
 case (FSNOTIFY_EVENT_PATH):
  inode = ((struct path *)data)->dentry->d_inode;
  break;
 case (FSNOTIFY_EVENT_INODE):
  inode = (struct inode *)data;
  break;
 default:
  BUG();
  return 0;
 };

 if (mask & (FS_CREATE|FS_MOVED_TO|FS_DELETE|FS_MOVED_FROM)) {
  if (audit_compare_dname_path(dname, audit_mark->path, AUDIT_NAME_FULL))
   return 0;
  audit_update_mark(audit_mark, inode);
 } else if (mask & (FS_DELETE_SELF|FS_UNMOUNT|FS_MOVE_SELF))
  audit_autoremove_mark_rule(audit_mark);

 return 0;
}

static const struct fsnotify_ops audit_mark_fsnotify_ops = {
 .handle_event = audit_mark_handle_event,
};

static int __init audit_fsnotify_init(void)
{
 audit_fsnotify_group = fsnotify_alloc_group(&audit_mark_fsnotify_ops);
 if (IS_ERR(audit_fsnotify_group)) {
  audit_fsnotify_group = NULL;
  audit_panic("cannot create audit fsnotify group");
 }
 return 0;
}
device_initcall(audit_fsnotify_init);










int audit_n_rules;


int audit_signals;

struct audit_aux_data {
 struct audit_aux_data *next;
 int type;
};




struct audit_aux_data_pids {
 struct audit_aux_data d;
 pid_t target_pid[AUDIT_AUX_PIDS];
 kuid_t target_auid[AUDIT_AUX_PIDS];
 kuid_t target_uid[AUDIT_AUX_PIDS];
 unsigned int target_sessionid[AUDIT_AUX_PIDS];
 u32 target_sid[AUDIT_AUX_PIDS];
 char target_comm[AUDIT_AUX_PIDS][TASK_COMM_LEN];
 int pid_count;
};

struct audit_aux_data_bprm_fcaps {
 struct audit_aux_data d;
 struct audit_cap_data fcap;
 unsigned int fcap_ver;
 struct audit_cap_data old_pcap;
 struct audit_cap_data new_pcap;
};

struct audit_tree_refs {
 struct audit_tree_refs *next;
 struct audit_chunk *c[31];
};

static int audit_match_perm(struct audit_context *ctx, int mask)
{
 unsigned n;
 if (unlikely(!ctx))
  return 0;
 n = ctx->major;

 switch (audit_classify_syscall(ctx->arch, n)) {
 case 0:
  if ((mask & AUDIT_PERM_WRITE) &&
       audit_match_class(AUDIT_CLASS_WRITE, n))
   return 1;
  if ((mask & AUDIT_PERM_READ) &&
       audit_match_class(AUDIT_CLASS_READ, n))
   return 1;
  if ((mask & AUDIT_PERM_ATTR) &&
       audit_match_class(AUDIT_CLASS_CHATTR, n))
   return 1;
  return 0;
 case 1:
  if ((mask & AUDIT_PERM_WRITE) &&
       audit_match_class(AUDIT_CLASS_WRITE_32, n))
   return 1;
  if ((mask & AUDIT_PERM_READ) &&
       audit_match_class(AUDIT_CLASS_READ_32, n))
   return 1;
  if ((mask & AUDIT_PERM_ATTR) &&
       audit_match_class(AUDIT_CLASS_CHATTR_32, n))
   return 1;
  return 0;
 case 2:
  return mask & ACC_MODE(ctx->argv[1]);
 case 3:
  return mask & ACC_MODE(ctx->argv[2]);
 case 4:
  return ((mask & AUDIT_PERM_WRITE) && ctx->argv[0] == SYS_BIND);
 case 5:
  return mask & AUDIT_PERM_EXEC;
 default:
  return 0;
 }
}

static int audit_match_filetype(struct audit_context *ctx, int val)
{
 struct audit_names *n;
 umode_t mode = (umode_t)val;

 if (unlikely(!ctx))
  return 0;

 list_for_each_entry(n, &ctx->names_list, list) {
  if ((n->ino != AUDIT_INO_UNSET) &&
      ((n->mode & S_IFMT) == mode))
   return 1;
 }

 return 0;
}
static void audit_set_auditable(struct audit_context *ctx)
{
 if (!ctx->prio) {
  ctx->prio = 1;
  ctx->current_state = AUDIT_RECORD_CONTEXT;
 }
}

static int put_tree_ref(struct audit_context *ctx, struct audit_chunk *chunk)
{
 struct audit_tree_refs *p = ctx->trees;
 int left = ctx->tree_count;
 if (likely(left)) {
  p->c[--left] = chunk;
  ctx->tree_count = left;
  return 1;
 }
 if (!p)
  return 0;
 p = p->next;
 if (p) {
  p->c[30] = chunk;
  ctx->trees = p;
  ctx->tree_count = 30;
  return 1;
 }
 return 0;
}

static int grow_tree_refs(struct audit_context *ctx)
{
 struct audit_tree_refs *p = ctx->trees;
 ctx->trees = kzalloc(sizeof(struct audit_tree_refs), GFP_KERNEL);
 if (!ctx->trees) {
  ctx->trees = p;
  return 0;
 }
 if (p)
  p->next = ctx->trees;
 else
  ctx->first_trees = ctx->trees;
 ctx->tree_count = 31;
 return 1;
}

static void unroll_tree_refs(struct audit_context *ctx,
        struct audit_tree_refs *p, int count)
{
 struct audit_tree_refs *q;
 int n;
 if (!p) {

  p = ctx->first_trees;
  count = 31;

  if (!p)
   return;
 }
 n = count;
 for (q = p; q != ctx->trees; q = q->next, n = 31) {
  while (n--) {
   audit_put_chunk(q->c[n]);
   q->c[n] = NULL;
  }
 }
 while (n-- > ctx->tree_count) {
  audit_put_chunk(q->c[n]);
  q->c[n] = NULL;
 }
 ctx->trees = p;
 ctx->tree_count = count;
}

static void free_tree_refs(struct audit_context *ctx)
{
 struct audit_tree_refs *p, *q;
 for (p = ctx->first_trees; p; p = q) {
  q = p->next;
  kfree(p);
 }
}

static int match_tree_refs(struct audit_context *ctx, struct audit_tree *tree)
{
 struct audit_tree_refs *p;
 int n;
 if (!tree)
  return 0;

 for (p = ctx->first_trees; p != ctx->trees; p = p->next) {
  for (n = 0; n < 31; n++)
   if (audit_tree_match(p->c[n], tree))
    return 1;
 }

 if (p) {
  for (n = ctx->tree_count; n < 31; n++)
   if (audit_tree_match(p->c[n], tree))
    return 1;
 }
 return 0;
}

static int audit_compare_uid(kuid_t uid,
        struct audit_names *name,
        struct audit_field *f,
        struct audit_context *ctx)
{
 struct audit_names *n;
 int rc;

 if (name) {
  rc = audit_uid_comparator(uid, f->op, name->uid);
  if (rc)
   return rc;
 }

 if (ctx) {
  list_for_each_entry(n, &ctx->names_list, list) {
   rc = audit_uid_comparator(uid, f->op, n->uid);
   if (rc)
    return rc;
  }
 }
 return 0;
}

static int audit_compare_gid(kgid_t gid,
        struct audit_names *name,
        struct audit_field *f,
        struct audit_context *ctx)
{
 struct audit_names *n;
 int rc;

 if (name) {
  rc = audit_gid_comparator(gid, f->op, name->gid);
  if (rc)
   return rc;
 }

 if (ctx) {
  list_for_each_entry(n, &ctx->names_list, list) {
   rc = audit_gid_comparator(gid, f->op, n->gid);
   if (rc)
    return rc;
  }
 }
 return 0;
}

static int audit_field_compare(struct task_struct *tsk,
          const struct cred *cred,
          struct audit_field *f,
          struct audit_context *ctx,
          struct audit_names *name)
{
 switch (f->val) {

 case AUDIT_COMPARE_UID_TO_OBJ_UID:
  return audit_compare_uid(cred->uid, name, f, ctx);
 case AUDIT_COMPARE_GID_TO_OBJ_GID:
  return audit_compare_gid(cred->gid, name, f, ctx);
 case AUDIT_COMPARE_EUID_TO_OBJ_UID:
  return audit_compare_uid(cred->euid, name, f, ctx);
 case AUDIT_COMPARE_EGID_TO_OBJ_GID:
  return audit_compare_gid(cred->egid, name, f, ctx);
 case AUDIT_COMPARE_AUID_TO_OBJ_UID:
  return audit_compare_uid(tsk->loginuid, name, f, ctx);
 case AUDIT_COMPARE_SUID_TO_OBJ_UID:
  return audit_compare_uid(cred->suid, name, f, ctx);
 case AUDIT_COMPARE_SGID_TO_OBJ_GID:
  return audit_compare_gid(cred->sgid, name, f, ctx);
 case AUDIT_COMPARE_FSUID_TO_OBJ_UID:
  return audit_compare_uid(cred->fsuid, name, f, ctx);
 case AUDIT_COMPARE_FSGID_TO_OBJ_GID:
  return audit_compare_gid(cred->fsgid, name, f, ctx);

 case AUDIT_COMPARE_UID_TO_AUID:
  return audit_uid_comparator(cred->uid, f->op, tsk->loginuid);
 case AUDIT_COMPARE_UID_TO_EUID:
  return audit_uid_comparator(cred->uid, f->op, cred->euid);
 case AUDIT_COMPARE_UID_TO_SUID:
  return audit_uid_comparator(cred->uid, f->op, cred->suid);
 case AUDIT_COMPARE_UID_TO_FSUID:
  return audit_uid_comparator(cred->uid, f->op, cred->fsuid);

 case AUDIT_COMPARE_AUID_TO_EUID:
  return audit_uid_comparator(tsk->loginuid, f->op, cred->euid);
 case AUDIT_COMPARE_AUID_TO_SUID:
  return audit_uid_comparator(tsk->loginuid, f->op, cred->suid);
 case AUDIT_COMPARE_AUID_TO_FSUID:
  return audit_uid_comparator(tsk->loginuid, f->op, cred->fsuid);

 case AUDIT_COMPARE_EUID_TO_SUID:
  return audit_uid_comparator(cred->euid, f->op, cred->suid);
 case AUDIT_COMPARE_EUID_TO_FSUID:
  return audit_uid_comparator(cred->euid, f->op, cred->fsuid);

 case AUDIT_COMPARE_SUID_TO_FSUID:
  return audit_uid_comparator(cred->suid, f->op, cred->fsuid);

 case AUDIT_COMPARE_GID_TO_EGID:
  return audit_gid_comparator(cred->gid, f->op, cred->egid);
 case AUDIT_COMPARE_GID_TO_SGID:
  return audit_gid_comparator(cred->gid, f->op, cred->sgid);
 case AUDIT_COMPARE_GID_TO_FSGID:
  return audit_gid_comparator(cred->gid, f->op, cred->fsgid);

 case AUDIT_COMPARE_EGID_TO_SGID:
  return audit_gid_comparator(cred->egid, f->op, cred->sgid);
 case AUDIT_COMPARE_EGID_TO_FSGID:
  return audit_gid_comparator(cred->egid, f->op, cred->fsgid);

 case AUDIT_COMPARE_SGID_TO_FSGID:
  return audit_gid_comparator(cred->sgid, f->op, cred->fsgid);
 default:
  WARN(1, "Missing AUDIT_COMPARE define.  Report as a bug\n");
  return 0;
 }
 return 0;
}
static int audit_filter_rules(struct task_struct *tsk,
         struct audit_krule *rule,
         struct audit_context *ctx,
         struct audit_names *name,
         enum audit_state *state,
         bool task_creation)
{
 const struct cred *cred;
 int i, need_sid = 1;
 u32 sid;

 cred = rcu_dereference_check(tsk->cred, tsk == current || task_creation);

 for (i = 0; i < rule->field_count; i++) {
  struct audit_field *f = &rule->fields[i];
  struct audit_names *n;
  int result = 0;
  pid_t pid;

  switch (f->type) {
  case AUDIT_PID:
   pid = task_pid_nr(tsk);
   result = audit_comparator(pid, f->op, f->val);
   break;
  case AUDIT_PPID:
   if (ctx) {
    if (!ctx->ppid)
     ctx->ppid = task_ppid_nr(tsk);
    result = audit_comparator(ctx->ppid, f->op, f->val);
   }
   break;
  case AUDIT_EXE:
   result = audit_exe_compare(tsk, rule->exe);
   break;
  case AUDIT_UID:
   result = audit_uid_comparator(cred->uid, f->op, f->uid);
   break;
  case AUDIT_EUID:
   result = audit_uid_comparator(cred->euid, f->op, f->uid);
   break;
  case AUDIT_SUID:
   result = audit_uid_comparator(cred->suid, f->op, f->uid);
   break;
  case AUDIT_FSUID:
   result = audit_uid_comparator(cred->fsuid, f->op, f->uid);
   break;
  case AUDIT_GID:
   result = audit_gid_comparator(cred->gid, f->op, f->gid);
   if (f->op == Audit_equal) {
    if (!result)
     result = in_group_p(f->gid);
   } else if (f->op == Audit_not_equal) {
    if (result)
     result = !in_group_p(f->gid);
   }
   break;
  case AUDIT_EGID:
   result = audit_gid_comparator(cred->egid, f->op, f->gid);
   if (f->op == Audit_equal) {
    if (!result)
     result = in_egroup_p(f->gid);
   } else if (f->op == Audit_not_equal) {
    if (result)
     result = !in_egroup_p(f->gid);
   }
   break;
  case AUDIT_SGID:
   result = audit_gid_comparator(cred->sgid, f->op, f->gid);
   break;
  case AUDIT_FSGID:
   result = audit_gid_comparator(cred->fsgid, f->op, f->gid);
   break;
  case AUDIT_PERS:
   result = audit_comparator(tsk->personality, f->op, f->val);
   break;
  case AUDIT_ARCH:
   if (ctx)
    result = audit_comparator(ctx->arch, f->op, f->val);
   break;

  case AUDIT_EXIT:
   if (ctx && ctx->return_valid)
    result = audit_comparator(ctx->return_code, f->op, f->val);
   break;
  case AUDIT_SUCCESS:
   if (ctx && ctx->return_valid) {
    if (f->val)
     result = audit_comparator(ctx->return_valid, f->op, AUDITSC_SUCCESS);
    else
     result = audit_comparator(ctx->return_valid, f->op, AUDITSC_FAILURE);
   }
   break;
  case AUDIT_DEVMAJOR:
   if (name) {
    if (audit_comparator(MAJOR(name->dev), f->op, f->val) ||
        audit_comparator(MAJOR(name->rdev), f->op, f->val))
     ++result;
   } else if (ctx) {
    list_for_each_entry(n, &ctx->names_list, list) {
     if (audit_comparator(MAJOR(n->dev), f->op, f->val) ||
         audit_comparator(MAJOR(n->rdev), f->op, f->val)) {
      ++result;
      break;
     }
    }
   }
   break;
  case AUDIT_DEVMINOR:
   if (name) {
    if (audit_comparator(MINOR(name->dev), f->op, f->val) ||
        audit_comparator(MINOR(name->rdev), f->op, f->val))
     ++result;
   } else if (ctx) {
    list_for_each_entry(n, &ctx->names_list, list) {
     if (audit_comparator(MINOR(n->dev), f->op, f->val) ||
         audit_comparator(MINOR(n->rdev), f->op, f->val)) {
      ++result;
      break;
     }
    }
   }
   break;
  case AUDIT_INODE:
   if (name)
    result = audit_comparator(name->ino, f->op, f->val);
   else if (ctx) {
    list_for_each_entry(n, &ctx->names_list, list) {
     if (audit_comparator(n->ino, f->op, f->val)) {
      ++result;
      break;
     }
    }
   }
   break;
  case AUDIT_OBJ_UID:
   if (name) {
    result = audit_uid_comparator(name->uid, f->op, f->uid);
   } else if (ctx) {
    list_for_each_entry(n, &ctx->names_list, list) {
     if (audit_uid_comparator(n->uid, f->op, f->uid)) {
      ++result;
      break;
     }
    }
   }
   break;
  case AUDIT_OBJ_GID:
   if (name) {
    result = audit_gid_comparator(name->gid, f->op, f->gid);
   } else if (ctx) {
    list_for_each_entry(n, &ctx->names_list, list) {
     if (audit_gid_comparator(n->gid, f->op, f->gid)) {
      ++result;
      break;
     }
    }
   }
   break;
  case AUDIT_WATCH:
   if (name)
    result = audit_watch_compare(rule->watch, name->ino, name->dev);
   break;
  case AUDIT_DIR:
   if (ctx)
    result = match_tree_refs(ctx, rule->tree);
   break;
  case AUDIT_LOGINUID:
   result = audit_uid_comparator(tsk->loginuid, f->op, f->uid);
   break;
  case AUDIT_LOGINUID_SET:
   result = audit_comparator(audit_loginuid_set(tsk), f->op, f->val);
   break;
  case AUDIT_SUBJ_USER:
  case AUDIT_SUBJ_ROLE:
  case AUDIT_SUBJ_TYPE:
  case AUDIT_SUBJ_SEN:
  case AUDIT_SUBJ_CLR:





   if (f->lsm_rule) {
    if (need_sid) {
     security_task_getsecid(tsk, &sid);
     need_sid = 0;
    }
    result = security_audit_rule_match(sid, f->type,
                                      f->op,
                                      f->lsm_rule,
                                      ctx);
   }
   break;
  case AUDIT_OBJ_USER:
  case AUDIT_OBJ_ROLE:
  case AUDIT_OBJ_TYPE:
  case AUDIT_OBJ_LEV_LOW:
  case AUDIT_OBJ_LEV_HIGH:


   if (f->lsm_rule) {

    if (name) {
     result = security_audit_rule_match(
                name->osid, f->type, f->op,
                f->lsm_rule, ctx);
    } else if (ctx) {
     list_for_each_entry(n, &ctx->names_list, list) {
      if (security_audit_rule_match(n->osid, f->type,
               f->op, f->lsm_rule,
               ctx)) {
       ++result;
       break;
      }
     }
    }

    if (!ctx || ctx->type != AUDIT_IPC)
     break;
    if (security_audit_rule_match(ctx->ipc.osid,
             f->type, f->op,
             f->lsm_rule, ctx))
     ++result;
   }
   break;
  case AUDIT_ARG0:
  case AUDIT_ARG1:
  case AUDIT_ARG2:
  case AUDIT_ARG3:
   if (ctx)
    result = audit_comparator(ctx->argv[f->type-AUDIT_ARG0], f->op, f->val);
   break;
  case AUDIT_FILTERKEY:

   result = 1;
   break;
  case AUDIT_PERM:
   result = audit_match_perm(ctx, f->val);
   break;
  case AUDIT_FILETYPE:
   result = audit_match_filetype(ctx, f->val);
   break;
  case AUDIT_FIELD_COMPARE:
   result = audit_field_compare(tsk, cred, f, ctx, name);
   break;
  }
  if (!result)
   return 0;
 }

 if (ctx) {
  if (rule->prio <= ctx->prio)
   return 0;
  if (rule->filterkey) {
   kfree(ctx->filterkey);
   ctx->filterkey = kstrdup(rule->filterkey, GFP_ATOMIC);
  }
  ctx->prio = rule->prio;
 }
 switch (rule->action) {
 case AUDIT_NEVER: *state = AUDIT_DISABLED; break;
 case AUDIT_ALWAYS: *state = AUDIT_RECORD_CONTEXT; break;
 }
 return 1;
}





static enum audit_state audit_filter_task(struct task_struct *tsk, char **key)
{
 struct audit_entry *e;
 enum audit_state state;

 rcu_read_lock();
 list_for_each_entry_rcu(e, &audit_filter_list[AUDIT_FILTER_TASK], list) {
  if (audit_filter_rules(tsk, &e->rule, NULL, NULL,
           &state, true)) {
   if (state == AUDIT_RECORD_CONTEXT)
    *key = kstrdup(e->rule.filterkey, GFP_ATOMIC);
   rcu_read_unlock();
   return state;
  }
 }
 rcu_read_unlock();
 return AUDIT_BUILD_CONTEXT;
}

static int audit_in_mask(const struct audit_krule *rule, unsigned long val)
{
 int word, bit;

 if (val > 0xffffffff)
  return false;

 word = AUDIT_WORD(val);
 if (word >= AUDIT_BITMASK_SIZE)
  return false;

 bit = AUDIT_BIT(val);

 return rule->mask[word] & bit;
}






static enum audit_state audit_filter_syscall(struct task_struct *tsk,
          struct audit_context *ctx,
          struct list_head *list)
{
 struct audit_entry *e;
 enum audit_state state;

 if (audit_pid && tsk->tgid == audit_pid)
  return AUDIT_DISABLED;

 rcu_read_lock();
 if (!list_empty(list)) {
  list_for_each_entry_rcu(e, list, list) {
   if (audit_in_mask(&e->rule, ctx->major) &&
       audit_filter_rules(tsk, &e->rule, ctx, NULL,
            &state, false)) {
    rcu_read_unlock();
    ctx->current_state = state;
    return state;
   }
  }
 }
 rcu_read_unlock();
 return AUDIT_BUILD_CONTEXT;
}





static int audit_filter_inode_name(struct task_struct *tsk,
       struct audit_names *n,
       struct audit_context *ctx) {
 int h = audit_hash_ino((u32)n->ino);
 struct list_head *list = &audit_inode_hash[h];
 struct audit_entry *e;
 enum audit_state state;

 if (list_empty(list))
  return 0;

 list_for_each_entry_rcu(e, list, list) {
  if (audit_in_mask(&e->rule, ctx->major) &&
      audit_filter_rules(tsk, &e->rule, ctx, n, &state, false)) {
   ctx->current_state = state;
   return 1;
  }
 }

 return 0;
}






void audit_filter_inodes(struct task_struct *tsk, struct audit_context *ctx)
{
 struct audit_names *n;

 if (audit_pid && tsk->tgid == audit_pid)
  return;

 rcu_read_lock();

 list_for_each_entry(n, &ctx->names_list, list) {
  if (audit_filter_inode_name(tsk, n, ctx))
   break;
 }
 rcu_read_unlock();
}


static inline struct audit_context *audit_take_context(struct task_struct *tsk,
            int return_valid,
            long return_code)
{
 struct audit_context *context = tsk->audit_context;

 if (!context)
  return NULL;
 context->return_valid = return_valid;
 if (unlikely(return_code <= -ERESTARTSYS) &&
     (return_code >= -ERESTART_RESTARTBLOCK) &&
     (return_code != -ENOIOCTLCMD))
  context->return_code = -EINTR;
 else
  context->return_code = return_code;

 if (context->in_syscall && !context->dummy) {
  audit_filter_syscall(tsk, context, &audit_filter_list[AUDIT_FILTER_EXIT]);
  audit_filter_inodes(tsk, context);
 }

 tsk->audit_context = NULL;
 return context;
}

static inline void audit_proctitle_free(struct audit_context *context)
{
 kfree(context->proctitle.value);
 context->proctitle.value = NULL;
 context->proctitle.len = 0;
}

static inline void audit_free_names(struct audit_context *context)
{
 struct audit_names *n, *next;

 list_for_each_entry_safe(n, next, &context->names_list, list) {
  list_del(&n->list);
  if (n->name)
   putname(n->name);
  if (n->should_free)
   kfree(n);
 }
 context->name_count = 0;
 path_put(&context->pwd);
 context->pwd.dentry = NULL;
 context->pwd.mnt = NULL;
}

static inline void audit_free_aux(struct audit_context *context)
{
 struct audit_aux_data *aux;

 while ((aux = context->aux)) {
  context->aux = aux->next;
  kfree(aux);
 }
 while ((aux = context->aux_pids)) {
  context->aux_pids = aux->next;
  kfree(aux);
 }
}

static inline struct audit_context *audit_alloc_context(enum audit_state state)
{
 struct audit_context *context;

 context = kzalloc(sizeof(*context), GFP_KERNEL);
 if (!context)
  return NULL;
 context->state = state;
 context->prio = state == AUDIT_RECORD_CONTEXT ? ~0ULL : 0;
 INIT_LIST_HEAD(&context->killed_trees);
 INIT_LIST_HEAD(&context->names_list);
 return context;
}
int audit_alloc(struct task_struct *tsk)
{
 struct audit_context *context;
 enum audit_state state;
 char *key = NULL;

 if (likely(!audit_ever_enabled))
  return 0;

 state = audit_filter_task(tsk, &key);
 if (state == AUDIT_DISABLED) {
  clear_tsk_thread_flag(tsk, TIF_SYSCALL_AUDIT);
  return 0;
 }

 if (!(context = audit_alloc_context(state))) {
  kfree(key);
  audit_log_lost("out of memory in audit_alloc");
  return -ENOMEM;
 }
 context->filterkey = key;

 tsk->audit_context = context;
 set_tsk_thread_flag(tsk, TIF_SYSCALL_AUDIT);
 return 0;
}

static inline void audit_free_context(struct audit_context *context)
{
 audit_free_names(context);
 unroll_tree_refs(context, NULL, 0);
 free_tree_refs(context);
 audit_free_aux(context);
 kfree(context->filterkey);
 kfree(context->sockaddr);
 audit_proctitle_free(context);
 kfree(context);
}

static int audit_log_pid_context(struct audit_context *context, pid_t pid,
     kuid_t auid, kuid_t uid, unsigned int sessionid,
     u32 sid, char *comm)
{
 struct audit_buffer *ab;
 char *ctx = NULL;
 u32 len;
 int rc = 0;

 ab = audit_log_start(context, GFP_KERNEL, AUDIT_OBJ_PID);
 if (!ab)
  return rc;

 audit_log_format(ab, "opid=%d oauid=%d ouid=%d oses=%d", pid,
    from_kuid(&init_user_ns, auid),
    from_kuid(&init_user_ns, uid), sessionid);
 if (sid) {
  if (security_secid_to_secctx(sid, &ctx, &len)) {
   audit_log_format(ab, " obj=(none)");
   rc = 1;
  } else {
   audit_log_format(ab, " obj=%s", ctx);
   security_release_secctx(ctx, len);
  }
 }
 audit_log_format(ab, " ocomm=");
 audit_log_untrustedstring(ab, comm);
 audit_log_end(ab);

 return rc;
}
static int audit_log_single_execve_arg(struct audit_context *context,
     struct audit_buffer **ab,
     int arg_num,
     size_t *len_sent,
     const char __user *p,
     char *buf)
{
 char arg_num_len_buf[12];
 const char __user *tmp_p = p;

 size_t arg_num_len = snprintf(arg_num_len_buf, 12, "%d", arg_num) + 5;
 size_t len, len_left, to_send;
 size_t max_execve_audit_len = MAX_EXECVE_AUDIT_LEN;
 unsigned int i, has_cntl = 0, too_long = 0;
 int ret;


 len_left = len = strnlen_user(p, MAX_ARG_STRLEN) - 1;







 if (WARN_ON_ONCE(len < 0 || len > MAX_ARG_STRLEN - 1)) {
  send_sig(SIGKILL, current, 0);
  return -1;
 }


 do {
  if (len_left > MAX_EXECVE_AUDIT_LEN)
   to_send = MAX_EXECVE_AUDIT_LEN;
  else
   to_send = len_left;
  ret = copy_from_user(buf, tmp_p, to_send);





  if (ret) {
   WARN_ON(1);
   send_sig(SIGKILL, current, 0);
   return -1;
  }
  buf[to_send] = '\0';
  has_cntl = audit_string_contains_control(buf, to_send);
  if (has_cntl) {




   max_execve_audit_len = MAX_EXECVE_AUDIT_LEN / 2;
   break;
  }
  len_left -= to_send;
  tmp_p += to_send;
 } while (len_left > 0);

 len_left = len;

 if (len > max_execve_audit_len)
  too_long = 1;


 for (i = 0; len_left > 0; i++) {
  int room_left;

  if (len_left > max_execve_audit_len)
   to_send = max_execve_audit_len;
  else
   to_send = len_left;


  room_left = MAX_EXECVE_AUDIT_LEN - arg_num_len - *len_sent;
  if (has_cntl)
   room_left -= (to_send * 2);
  else
   room_left -= to_send;
  if (room_left < 0) {
   *len_sent = 0;
   audit_log_end(*ab);
   *ab = audit_log_start(context, GFP_KERNEL, AUDIT_EXECVE);
   if (!*ab)
    return 0;
  }





  if ((i == 0) && (too_long))
   audit_log_format(*ab, " a%d_len=%zu", arg_num,
      has_cntl ? 2*len : len);






  if (len >= max_execve_audit_len)
   ret = copy_from_user(buf, p, to_send);
  else
   ret = 0;
  if (ret) {
   WARN_ON(1);
   send_sig(SIGKILL, current, 0);
   return -1;
  }
  buf[to_send] = '\0';


  audit_log_format(*ab, " a%d", arg_num);
  if (too_long)
   audit_log_format(*ab, "[%d]", i);
  audit_log_format(*ab, "=");
  if (has_cntl)
   audit_log_n_hex(*ab, buf, to_send);
  else
   audit_log_string(*ab, buf);

  p += to_send;
  len_left -= to_send;
  *len_sent += arg_num_len;
  if (has_cntl)
   *len_sent += to_send * 2;
  else
   *len_sent += to_send;
 }

 return len + 1;
}

static void audit_log_execve_info(struct audit_context *context,
      struct audit_buffer **ab)
{
 int i, len;
 size_t len_sent = 0;
 const char __user *p;
 char *buf;

 p = (const char __user *)current->mm->arg_start;

 audit_log_format(*ab, "argc=%d", context->execve.argc);







 buf = kmalloc(MAX_EXECVE_AUDIT_LEN + 1, GFP_KERNEL);
 if (!buf) {
  audit_panic("out of memory for argv string");
  return;
 }

 for (i = 0; i < context->execve.argc; i++) {
  len = audit_log_single_execve_arg(context, ab, i,
        &len_sent, p, buf);
  if (len <= 0)
   break;
  p += len;
 }
 kfree(buf);
}

static void show_special(struct audit_context *context, int *call_panic)
{
 struct audit_buffer *ab;
 int i;

 ab = audit_log_start(context, GFP_KERNEL, context->type);
 if (!ab)
  return;

 switch (context->type) {
 case AUDIT_SOCKETCALL: {
  int nargs = context->socketcall.nargs;
  audit_log_format(ab, "nargs=%d", nargs);
  for (i = 0; i < nargs; i++)
   audit_log_format(ab, " a%d=%lx", i,
    context->socketcall.args[i]);
  break; }
 case AUDIT_IPC: {
  u32 osid = context->ipc.osid;

  audit_log_format(ab, "ouid=%u ogid=%u mode=%#ho",
     from_kuid(&init_user_ns, context->ipc.uid),
     from_kgid(&init_user_ns, context->ipc.gid),
     context->ipc.mode);
  if (osid) {
   char *ctx = NULL;
   u32 len;
   if (security_secid_to_secctx(osid, &ctx, &len)) {
    audit_log_format(ab, " osid=%u", osid);
    *call_panic = 1;
   } else {
    audit_log_format(ab, " obj=%s", ctx);
    security_release_secctx(ctx, len);
   }
  }
  if (context->ipc.has_perm) {
   audit_log_end(ab);
   ab = audit_log_start(context, GFP_KERNEL,
          AUDIT_IPC_SET_PERM);
   if (unlikely(!ab))
    return;
   audit_log_format(ab,
    "qbytes=%lx ouid=%u ogid=%u mode=%#ho",
    context->ipc.qbytes,
    context->ipc.perm_uid,
    context->ipc.perm_gid,
    context->ipc.perm_mode);
  }
  break; }
 case AUDIT_MQ_OPEN: {
  audit_log_format(ab,
   "oflag=0x%x mode=%#ho mq_flags=0x%lx mq_maxmsg=%ld "
   "mq_msgsize=%ld mq_curmsgs=%ld",
   context->mq_open.oflag, context->mq_open.mode,
   context->mq_open.attr.mq_flags,
   context->mq_open.attr.mq_maxmsg,
   context->mq_open.attr.mq_msgsize,
   context->mq_open.attr.mq_curmsgs);
  break; }
 case AUDIT_MQ_SENDRECV: {
  audit_log_format(ab,
   "mqdes=%d msg_len=%zd msg_prio=%u "
   "abs_timeout_sec=%ld abs_timeout_nsec=%ld",
   context->mq_sendrecv.mqdes,
   context->mq_sendrecv.msg_len,
   context->mq_sendrecv.msg_prio,
   context->mq_sendrecv.abs_timeout.tv_sec,
   context->mq_sendrecv.abs_timeout.tv_nsec);
  break; }
 case AUDIT_MQ_NOTIFY: {
  audit_log_format(ab, "mqdes=%d sigev_signo=%d",
    context->mq_notify.mqdes,
    context->mq_notify.sigev_signo);
  break; }
 case AUDIT_MQ_GETSETATTR: {
  struct mq_attr *attr = &context->mq_getsetattr.mqstat;
  audit_log_format(ab,
   "mqdes=%d mq_flags=0x%lx mq_maxmsg=%ld mq_msgsize=%ld "
   "mq_curmsgs=%ld ",
   context->mq_getsetattr.mqdes,
   attr->mq_flags, attr->mq_maxmsg,
   attr->mq_msgsize, attr->mq_curmsgs);
  break; }
 case AUDIT_CAPSET: {
  audit_log_format(ab, "pid=%d", context->capset.pid);
  audit_log_cap(ab, "cap_pi", &context->capset.cap.inheritable);
  audit_log_cap(ab, "cap_pp", &context->capset.cap.permitted);
  audit_log_cap(ab, "cap_pe", &context->capset.cap.effective);
  break; }
 case AUDIT_MMAP: {
  audit_log_format(ab, "fd=%d flags=0x%x", context->mmap.fd,
     context->mmap.flags);
  break; }
 case AUDIT_EXECVE: {
  audit_log_execve_info(context, &ab);
  break; }
 }
 audit_log_end(ab);
}

static inline int audit_proctitle_rtrim(char *proctitle, int len)
{
 char *end = proctitle + len - 1;
 while (end > proctitle && !isprint(*end))
  end--;


 len = end - proctitle + 1;
 len -= isprint(proctitle[len-1]) == 0;
 return len;
}

static void audit_log_proctitle(struct task_struct *tsk,
    struct audit_context *context)
{
 int res;
 char *buf;
 char *msg = "(null)";
 int len = strlen(msg);
 struct audit_buffer *ab;

 ab = audit_log_start(context, GFP_KERNEL, AUDIT_PROCTITLE);
 if (!ab)
  return;

 audit_log_format(ab, "proctitle=");


 if (!context->proctitle.value) {
  buf = kmalloc(MAX_PROCTITLE_AUDIT_LEN, GFP_KERNEL);
  if (!buf)
   goto out;

  res = get_cmdline(tsk, buf, MAX_PROCTITLE_AUDIT_LEN);
  if (res == 0) {
   kfree(buf);
   goto out;
  }
  res = audit_proctitle_rtrim(buf, res);
  if (res == 0) {
   kfree(buf);
   goto out;
  }
  context->proctitle.value = buf;
  context->proctitle.len = res;
 }
 msg = context->proctitle.value;
 len = context->proctitle.len;
out:
 audit_log_n_untrustedstring(ab, msg, len);
 audit_log_end(ab);
}

static void audit_log_exit(struct audit_context *context, struct task_struct *tsk)
{
 int i, call_panic = 0;
 struct audit_buffer *ab;
 struct audit_aux_data *aux;
 struct audit_names *n;


 context->personality = tsk->personality;

 ab = audit_log_start(context, GFP_KERNEL, AUDIT_SYSCALL);
 if (!ab)
  return;
 audit_log_format(ab, "arch=%x syscall=%d",
    context->arch, context->major);
 if (context->personality != PER_LINUX)
  audit_log_format(ab, " per=%lx", context->personality);
 if (context->return_valid)
  audit_log_format(ab, " success=%s exit=%ld",
     (context->return_valid==AUDITSC_SUCCESS)?"yes":"no",
     context->return_code);

 audit_log_format(ab,
    " a0=%lx a1=%lx a2=%lx a3=%lx items=%d",
    context->argv[0],
    context->argv[1],
    context->argv[2],
    context->argv[3],
    context->name_count);

 audit_log_task_info(ab, tsk);
 audit_log_key(ab, context->filterkey);
 audit_log_end(ab);

 for (aux = context->aux; aux; aux = aux->next) {

  ab = audit_log_start(context, GFP_KERNEL, aux->type);
  if (!ab)
   continue;

  switch (aux->type) {

  case AUDIT_BPRM_FCAPS: {
   struct audit_aux_data_bprm_fcaps *axs = (void *)aux;
   audit_log_format(ab, "fver=%x", axs->fcap_ver);
   audit_log_cap(ab, "fp", &axs->fcap.permitted);
   audit_log_cap(ab, "fi", &axs->fcap.inheritable);
   audit_log_format(ab, " fe=%d", axs->fcap.fE);
   audit_log_cap(ab, "old_pp", &axs->old_pcap.permitted);
   audit_log_cap(ab, "old_pi", &axs->old_pcap.inheritable);
   audit_log_cap(ab, "old_pe", &axs->old_pcap.effective);
   audit_log_cap(ab, "new_pp", &axs->new_pcap.permitted);
   audit_log_cap(ab, "new_pi", &axs->new_pcap.inheritable);
   audit_log_cap(ab, "new_pe", &axs->new_pcap.effective);
   break; }

  }
  audit_log_end(ab);
 }

 if (context->type)
  show_special(context, &call_panic);

 if (context->fds[0] >= 0) {
  ab = audit_log_start(context, GFP_KERNEL, AUDIT_FD_PAIR);
  if (ab) {
   audit_log_format(ab, "fd0=%d fd1=%d",
     context->fds[0], context->fds[1]);
   audit_log_end(ab);
  }
 }

 if (context->sockaddr_len) {
  ab = audit_log_start(context, GFP_KERNEL, AUDIT_SOCKADDR);
  if (ab) {
   audit_log_format(ab, "saddr=");
   audit_log_n_hex(ab, (void *)context->sockaddr,
     context->sockaddr_len);
   audit_log_end(ab);
  }
 }

 for (aux = context->aux_pids; aux; aux = aux->next) {
  struct audit_aux_data_pids *axs = (void *)aux;

  for (i = 0; i < axs->pid_count; i++)
   if (audit_log_pid_context(context, axs->target_pid[i],
        axs->target_auid[i],
        axs->target_uid[i],
        axs->target_sessionid[i],
        axs->target_sid[i],
        axs->target_comm[i]))
    call_panic = 1;
 }

 if (context->target_pid &&
     audit_log_pid_context(context, context->target_pid,
      context->target_auid, context->target_uid,
      context->target_sessionid,
      context->target_sid, context->target_comm))
   call_panic = 1;

 if (context->pwd.dentry && context->pwd.mnt) {
  ab = audit_log_start(context, GFP_KERNEL, AUDIT_CWD);
  if (ab) {
   audit_log_d_path(ab, " cwd=", &context->pwd);
   audit_log_end(ab);
  }
 }

 i = 0;
 list_for_each_entry(n, &context->names_list, list) {
  if (n->hidden)
   continue;
  audit_log_name(context, n, NULL, i++, &call_panic);
 }

 audit_log_proctitle(tsk, context);


 ab = audit_log_start(context, GFP_KERNEL, AUDIT_EOE);
 if (ab)
  audit_log_end(ab);
 if (call_panic)
  audit_panic("error converting sid to string");
}







void __audit_free(struct task_struct *tsk)
{
 struct audit_context *context;

 context = audit_take_context(tsk, 0, 0);
 if (!context)
  return;






 if (context->in_syscall && context->current_state == AUDIT_RECORD_CONTEXT)
  audit_log_exit(context, tsk);
 if (!list_empty(&context->killed_trees))
  audit_kill_trees(&context->killed_trees);

 audit_free_context(context);
}
void __audit_syscall_entry(int major, unsigned long a1, unsigned long a2,
      unsigned long a3, unsigned long a4)
{
 struct task_struct *tsk = current;
 struct audit_context *context = tsk->audit_context;
 enum audit_state state;

 if (!context)
  return;

 BUG_ON(context->in_syscall || context->name_count);

 if (!audit_enabled)
  return;

 context->arch = syscall_get_arch();
 context->major = major;
 context->argv[0] = a1;
 context->argv[1] = a2;
 context->argv[2] = a3;
 context->argv[3] = a4;

 state = context->state;
 context->dummy = !audit_n_rules;
 if (!context->dummy && state == AUDIT_BUILD_CONTEXT) {
  context->prio = 0;
  state = audit_filter_syscall(tsk, context, &audit_filter_list[AUDIT_FILTER_ENTRY]);
 }
 if (state == AUDIT_DISABLED)
  return;

 context->serial = 0;
 context->ctime = CURRENT_TIME;
 context->in_syscall = 1;
 context->current_state = state;
 context->ppid = 0;
}
void __audit_syscall_exit(int success, long return_code)
{
 struct task_struct *tsk = current;
 struct audit_context *context;

 if (success)
  success = AUDITSC_SUCCESS;
 else
  success = AUDITSC_FAILURE;

 context = audit_take_context(tsk, success, return_code);
 if (!context)
  return;

 if (context->in_syscall && context->current_state == AUDIT_RECORD_CONTEXT)
  audit_log_exit(context, tsk);

 context->in_syscall = 0;
 context->prio = context->state == AUDIT_RECORD_CONTEXT ? ~0ULL : 0;

 if (!list_empty(&context->killed_trees))
  audit_kill_trees(&context->killed_trees);

 audit_free_names(context);
 unroll_tree_refs(context, NULL, 0);
 audit_free_aux(context);
 context->aux = NULL;
 context->aux_pids = NULL;
 context->target_pid = 0;
 context->target_sid = 0;
 context->sockaddr_len = 0;
 context->type = 0;
 context->fds[0] = -1;
 if (context->state != AUDIT_RECORD_CONTEXT) {
  kfree(context->filterkey);
  context->filterkey = NULL;
 }
 tsk->audit_context = context;
}

static inline void handle_one(const struct inode *inode)
{
 struct audit_context *context;
 struct audit_tree_refs *p;
 struct audit_chunk *chunk;
 int count;
 if (likely(hlist_empty(&inode->i_fsnotify_marks)))
  return;
 context = current->audit_context;
 p = context->trees;
 count = context->tree_count;
 rcu_read_lock();
 chunk = audit_tree_lookup(inode);
 rcu_read_unlock();
 if (!chunk)
  return;
 if (likely(put_tree_ref(context, chunk)))
  return;
 if (unlikely(!grow_tree_refs(context))) {
  pr_warn("out of memory, audit has lost a tree reference\n");
  audit_set_auditable(context);
  audit_put_chunk(chunk);
  unroll_tree_refs(context, p, count);
  return;
 }
 put_tree_ref(context, chunk);
}

static void handle_path(const struct dentry *dentry)
{
 struct audit_context *context;
 struct audit_tree_refs *p;
 const struct dentry *d, *parent;
 struct audit_chunk *drop;
 unsigned long seq;
 int count;

 context = current->audit_context;
 p = context->trees;
 count = context->tree_count;
retry:
 drop = NULL;
 d = dentry;
 rcu_read_lock();
 seq = read_seqbegin(&rename_lock);
 for(;;) {
  struct inode *inode = d_backing_inode(d);
  if (inode && unlikely(!hlist_empty(&inode->i_fsnotify_marks))) {
   struct audit_chunk *chunk;
   chunk = audit_tree_lookup(inode);
   if (chunk) {
    if (unlikely(!put_tree_ref(context, chunk))) {
     drop = chunk;
     break;
    }
   }
  }
  parent = d->d_parent;
  if (parent == d)
   break;
  d = parent;
 }
 if (unlikely(read_seqretry(&rename_lock, seq) || drop)) {
  rcu_read_unlock();
  if (!drop) {

   unroll_tree_refs(context, p, count);
   goto retry;
  }
  audit_put_chunk(drop);
  if (grow_tree_refs(context)) {

   unroll_tree_refs(context, p, count);
   goto retry;
  }

  pr_warn("out of memory, audit has lost a tree reference\n");
  unroll_tree_refs(context, p, count);
  audit_set_auditable(context);
  return;
 }
 rcu_read_unlock();
}

static struct audit_names *audit_alloc_name(struct audit_context *context,
      unsigned char type)
{
 struct audit_names *aname;

 if (context->name_count < AUDIT_NAMES) {
  aname = &context->preallocated_names[context->name_count];
  memset(aname, 0, sizeof(*aname));
 } else {
  aname = kzalloc(sizeof(*aname), GFP_NOFS);
  if (!aname)
   return NULL;
  aname->should_free = true;
 }

 aname->ino = AUDIT_INO_UNSET;
 aname->type = type;
 list_add_tail(&aname->list, &context->names_list);

 context->name_count++;
 return aname;
}
struct filename *
__audit_reusename(const __user char *uptr)
{
 struct audit_context *context = current->audit_context;
 struct audit_names *n;

 list_for_each_entry(n, &context->names_list, list) {
  if (!n->name)
   continue;
  if (n->name->uptr == uptr) {
   n->name->refcnt++;
   return n->name;
  }
 }
 return NULL;
}
void __audit_getname(struct filename *name)
{
 struct audit_context *context = current->audit_context;
 struct audit_names *n;

 if (!context->in_syscall)
  return;

 n = audit_alloc_name(context, AUDIT_TYPE_UNKNOWN);
 if (!n)
  return;

 n->name = name;
 n->name_len = AUDIT_NAME_FULL;
 name->aname = n;
 name->refcnt++;

 if (!context->pwd.dentry)
  get_fs_pwd(current->fs, &context->pwd);
}







void __audit_inode(struct filename *name, const struct dentry *dentry,
     unsigned int flags)
{
 struct audit_context *context = current->audit_context;
 struct inode *inode = d_backing_inode(dentry);
 struct audit_names *n;
 bool parent = flags & AUDIT_INODE_PARENT;

 if (!context->in_syscall)
  return;

 if (!name)
  goto out_alloc;





 n = name->aname;
 if (n) {
  if (parent) {
   if (n->type == AUDIT_TYPE_PARENT ||
       n->type == AUDIT_TYPE_UNKNOWN)
    goto out;
  } else {
   if (n->type != AUDIT_TYPE_PARENT)
    goto out;
  }
 }

 list_for_each_entry_reverse(n, &context->names_list, list) {
  if (n->ino) {

   if (n->ino != inode->i_ino ||
       n->dev != inode->i_sb->s_dev)
    continue;
  } else if (n->name) {

   if (strcmp(n->name->name, name->name))
    continue;
  } else

   continue;


  if (parent) {
   if (n->type == AUDIT_TYPE_PARENT ||
       n->type == AUDIT_TYPE_UNKNOWN)
    goto out;
  } else {
   if (n->type != AUDIT_TYPE_PARENT)
    goto out;
  }
 }

out_alloc:

 n = audit_alloc_name(context, AUDIT_TYPE_UNKNOWN);
 if (!n)
  return;
 if (name) {
  n->name = name;
  name->refcnt++;
 }

out:
 if (parent) {
  n->name_len = n->name ? parent_len(n->name->name) : AUDIT_NAME_FULL;
  n->type = AUDIT_TYPE_PARENT;
  if (flags & AUDIT_INODE_HIDDEN)
   n->hidden = true;
 } else {
  n->name_len = AUDIT_NAME_FULL;
  n->type = AUDIT_TYPE_NORMAL;
 }
 handle_path(dentry);
 audit_copy_inode(n, dentry, inode);
}

void __audit_file(const struct file *file)
{
 __audit_inode(NULL, file->f_path.dentry, 0);
}
void __audit_inode_child(struct inode *parent,
    const struct dentry *dentry,
    const unsigned char type)
{
 struct audit_context *context = current->audit_context;
 struct inode *inode = d_backing_inode(dentry);
 const char *dname = dentry->d_name.name;
 struct audit_names *n, *found_parent = NULL, *found_child = NULL;

 if (!context->in_syscall)
  return;

 if (inode)
  handle_one(inode);


 list_for_each_entry(n, &context->names_list, list) {
  if (!n->name ||
      (n->type != AUDIT_TYPE_PARENT &&
       n->type != AUDIT_TYPE_UNKNOWN))
   continue;

  if (n->ino == parent->i_ino && n->dev == parent->i_sb->s_dev &&
      !audit_compare_dname_path(dname,
           n->name->name, n->name_len)) {
   if (n->type == AUDIT_TYPE_UNKNOWN)
    n->type = AUDIT_TYPE_PARENT;
   found_parent = n;
   break;
  }
 }


 list_for_each_entry(n, &context->names_list, list) {

  if (!n->name ||
      (n->type != type && n->type != AUDIT_TYPE_UNKNOWN))
   continue;

  if (!strcmp(dname, n->name->name) ||
      !audit_compare_dname_path(dname, n->name->name,
      found_parent ?
      found_parent->name_len :
      AUDIT_NAME_FULL)) {
   if (n->type == AUDIT_TYPE_UNKNOWN)
    n->type = type;
   found_child = n;
   break;
  }
 }

 if (!found_parent) {

  n = audit_alloc_name(context, AUDIT_TYPE_PARENT);
  if (!n)
   return;
  audit_copy_inode(n, NULL, parent);
 }

 if (!found_child) {
  found_child = audit_alloc_name(context, type);
  if (!found_child)
   return;




  if (found_parent) {
   found_child->name = found_parent->name;
   found_child->name_len = AUDIT_NAME_FULL;
   found_child->name->refcnt++;
  }
 }

 if (inode)
  audit_copy_inode(found_child, dentry, inode);
 else
  found_child->ino = AUDIT_INO_UNSET;
}
EXPORT_SYMBOL_GPL(__audit_inode_child);
int auditsc_get_stamp(struct audit_context *ctx,
         struct timespec *t, unsigned int *serial)
{
 if (!ctx->in_syscall)
  return 0;
 if (!ctx->serial)
  ctx->serial = audit_serial();
 t->tv_sec = ctx->ctime.tv_sec;
 t->tv_nsec = ctx->ctime.tv_nsec;
 *serial = ctx->serial;
 if (!ctx->prio) {
  ctx->prio = 1;
  ctx->current_state = AUDIT_RECORD_CONTEXT;
 }
 return 1;
}


static atomic_t session_id = ATOMIC_INIT(0);

static int audit_set_loginuid_perm(kuid_t loginuid)
{

 if (!audit_loginuid_set(current))
  return 0;

 if (is_audit_feature_set(AUDIT_FEATURE_LOGINUID_IMMUTABLE))
  return -EPERM;

 if (!capable(CAP_AUDIT_CONTROL))
  return -EPERM;

 if (is_audit_feature_set(AUDIT_FEATURE_ONLY_UNSET_LOGINUID) && uid_valid(loginuid))
  return -EPERM;
 return 0;
}

static void audit_log_set_loginuid(kuid_t koldloginuid, kuid_t kloginuid,
       unsigned int oldsessionid, unsigned int sessionid,
       int rc)
{
 struct audit_buffer *ab;
 uid_t uid, oldloginuid, loginuid;
 struct tty_struct *tty;

 if (!audit_enabled)
  return;

 ab = audit_log_start(NULL, GFP_KERNEL, AUDIT_LOGIN);
 if (!ab)
  return;

 uid = from_kuid(&init_user_ns, task_uid(current));
 oldloginuid = from_kuid(&init_user_ns, koldloginuid);
 loginuid = from_kuid(&init_user_ns, kloginuid),
 tty = audit_get_tty(current);

 audit_log_format(ab, "pid=%d uid=%u", task_pid_nr(current), uid);
 audit_log_task_context(ab);
 audit_log_format(ab, " old-auid=%u auid=%u tty=%s old-ses=%u ses=%u res=%d",
    oldloginuid, loginuid, tty ? tty_name(tty) : "(none)",
    oldsessionid, sessionid, !rc);
 audit_put_tty(tty);
 audit_log_end(ab);
}
int audit_set_loginuid(kuid_t loginuid)
{
 struct task_struct *task = current;
 unsigned int oldsessionid, sessionid = (unsigned int)-1;
 kuid_t oldloginuid;
 int rc;

 oldloginuid = audit_get_loginuid(current);
 oldsessionid = audit_get_sessionid(current);

 rc = audit_set_loginuid_perm(loginuid);
 if (rc)
  goto out;


 if (uid_valid(loginuid))
  sessionid = (unsigned int)atomic_inc_return(&session_id);

 task->sessionid = sessionid;
 task->loginuid = loginuid;
out:
 audit_log_set_loginuid(oldloginuid, loginuid, oldsessionid, sessionid, rc);
 return rc;
}
void __audit_mq_open(int oflag, umode_t mode, struct mq_attr *attr)
{
 struct audit_context *context = current->audit_context;

 if (attr)
  memcpy(&context->mq_open.attr, attr, sizeof(struct mq_attr));
 else
  memset(&context->mq_open.attr, 0, sizeof(struct mq_attr));

 context->mq_open.oflag = oflag;
 context->mq_open.mode = mode;

 context->type = AUDIT_MQ_OPEN;
}
void __audit_mq_sendrecv(mqd_t mqdes, size_t msg_len, unsigned int msg_prio,
   const struct timespec *abs_timeout)
{
 struct audit_context *context = current->audit_context;
 struct timespec *p = &context->mq_sendrecv.abs_timeout;

 if (abs_timeout)
  memcpy(p, abs_timeout, sizeof(struct timespec));
 else
  memset(p, 0, sizeof(struct timespec));

 context->mq_sendrecv.mqdes = mqdes;
 context->mq_sendrecv.msg_len = msg_len;
 context->mq_sendrecv.msg_prio = msg_prio;

 context->type = AUDIT_MQ_SENDRECV;
}
void __audit_mq_notify(mqd_t mqdes, const struct sigevent *notification)
{
 struct audit_context *context = current->audit_context;

 if (notification)
  context->mq_notify.sigev_signo = notification->sigev_signo;
 else
  context->mq_notify.sigev_signo = 0;

 context->mq_notify.mqdes = mqdes;
 context->type = AUDIT_MQ_NOTIFY;
}







void __audit_mq_getsetattr(mqd_t mqdes, struct mq_attr *mqstat)
{
 struct audit_context *context = current->audit_context;
 context->mq_getsetattr.mqdes = mqdes;
 context->mq_getsetattr.mqstat = *mqstat;
 context->type = AUDIT_MQ_GETSETATTR;
}






void __audit_ipc_obj(struct kern_ipc_perm *ipcp)
{
 struct audit_context *context = current->audit_context;
 context->ipc.uid = ipcp->uid;
 context->ipc.gid = ipcp->gid;
 context->ipc.mode = ipcp->mode;
 context->ipc.has_perm = 0;
 security_ipc_getsecid(ipcp, &context->ipc.osid);
 context->type = AUDIT_IPC;
}
void __audit_ipc_set_perm(unsigned long qbytes, uid_t uid, gid_t gid, umode_t mode)
{
 struct audit_context *context = current->audit_context;

 context->ipc.qbytes = qbytes;
 context->ipc.perm_uid = uid;
 context->ipc.perm_gid = gid;
 context->ipc.perm_mode = mode;
 context->ipc.has_perm = 1;
}

void __audit_bprm(struct linux_binprm *bprm)
{
 struct audit_context *context = current->audit_context;

 context->type = AUDIT_EXECVE;
 context->execve.argc = bprm->argc;
}
int __audit_socketcall(int nargs, unsigned long *args)
{
 struct audit_context *context = current->audit_context;

 if (nargs <= 0 || nargs > AUDITSC_ARGS || !args)
  return -EINVAL;
 context->type = AUDIT_SOCKETCALL;
 context->socketcall.nargs = nargs;
 memcpy(context->socketcall.args, args, nargs * sizeof(unsigned long));
 return 0;
}







void __audit_fd_pair(int fd1, int fd2)
{
 struct audit_context *context = current->audit_context;
 context->fds[0] = fd1;
 context->fds[1] = fd2;
}
int __audit_sockaddr(int len, void *a)
{
 struct audit_context *context = current->audit_context;

 if (!context->sockaddr) {
  void *p = kmalloc(sizeof(struct sockaddr_storage), GFP_KERNEL);
  if (!p)
   return -ENOMEM;
  context->sockaddr = p;
 }

 context->sockaddr_len = len;
 memcpy(context->sockaddr, a, len);
 return 0;
}

void __audit_ptrace(struct task_struct *t)
{
 struct audit_context *context = current->audit_context;

 context->target_pid = task_pid_nr(t);
 context->target_auid = audit_get_loginuid(t);
 context->target_uid = task_uid(t);
 context->target_sessionid = audit_get_sessionid(t);
 security_task_getsecid(t, &context->target_sid);
 memcpy(context->target_comm, t->comm, TASK_COMM_LEN);
}
int __audit_signal_info(int sig, struct task_struct *t)
{
 struct audit_aux_data_pids *axp;
 struct task_struct *tsk = current;
 struct audit_context *ctx = tsk->audit_context;
 kuid_t uid = current_uid(), t_uid = task_uid(t);

 if (audit_pid && t->tgid == audit_pid) {
  if (sig == SIGTERM || sig == SIGHUP || sig == SIGUSR1 || sig == SIGUSR2) {
   audit_sig_pid = task_pid_nr(tsk);
   if (uid_valid(tsk->loginuid))
    audit_sig_uid = tsk->loginuid;
   else
    audit_sig_uid = uid;
   security_task_getsecid(tsk, &audit_sig_sid);
  }
  if (!audit_signals || audit_dummy_context())
   return 0;
 }



 if (!ctx->target_pid) {
  ctx->target_pid = task_tgid_nr(t);
  ctx->target_auid = audit_get_loginuid(t);
  ctx->target_uid = t_uid;
  ctx->target_sessionid = audit_get_sessionid(t);
  security_task_getsecid(t, &ctx->target_sid);
  memcpy(ctx->target_comm, t->comm, TASK_COMM_LEN);
  return 0;
 }

 axp = (void *)ctx->aux_pids;
 if (!axp || axp->pid_count == AUDIT_AUX_PIDS) {
  axp = kzalloc(sizeof(*axp), GFP_ATOMIC);
  if (!axp)
   return -ENOMEM;

  axp->d.type = AUDIT_OBJ_PID;
  axp->d.next = ctx->aux_pids;
  ctx->aux_pids = (void *)axp;
 }
 BUG_ON(axp->pid_count >= AUDIT_AUX_PIDS);

 axp->target_pid[axp->pid_count] = task_tgid_nr(t);
 axp->target_auid[axp->pid_count] = audit_get_loginuid(t);
 axp->target_uid[axp->pid_count] = t_uid;
 axp->target_sessionid[axp->pid_count] = audit_get_sessionid(t);
 security_task_getsecid(t, &axp->target_sid[axp->pid_count]);
 memcpy(axp->target_comm[axp->pid_count], t->comm, TASK_COMM_LEN);
 axp->pid_count++;

 return 0;
}
int __audit_log_bprm_fcaps(struct linux_binprm *bprm,
      const struct cred *new, const struct cred *old)
{
 struct audit_aux_data_bprm_fcaps *ax;
 struct audit_context *context = current->audit_context;
 struct cpu_vfs_cap_data vcaps;

 ax = kmalloc(sizeof(*ax), GFP_KERNEL);
 if (!ax)
  return -ENOMEM;

 ax->d.type = AUDIT_BPRM_FCAPS;
 ax->d.next = context->aux;
 context->aux = (void *)ax;

 get_vfs_caps_from_disk(bprm->file->f_path.dentry, &vcaps);

 ax->fcap.permitted = vcaps.permitted;
 ax->fcap.inheritable = vcaps.inheritable;
 ax->fcap.fE = !!(vcaps.magic_etc & VFS_CAP_FLAGS_EFFECTIVE);
 ax->fcap_ver = (vcaps.magic_etc & VFS_CAP_REVISION_MASK) >> VFS_CAP_REVISION_SHIFT;

 ax->old_pcap.permitted = old->cap_permitted;
 ax->old_pcap.inheritable = old->cap_inheritable;
 ax->old_pcap.effective = old->cap_effective;

 ax->new_pcap.permitted = new->cap_permitted;
 ax->new_pcap.inheritable = new->cap_inheritable;
 ax->new_pcap.effective = new->cap_effective;
 return 0;
}
void __audit_log_capset(const struct cred *new, const struct cred *old)
{
 struct audit_context *context = current->audit_context;
 context->capset.pid = task_pid_nr(current);
 context->capset.cap.effective = new->cap_effective;
 context->capset.cap.inheritable = new->cap_effective;
 context->capset.cap.permitted = new->cap_permitted;
 context->type = AUDIT_CAPSET;
}

void __audit_mmap_fd(int fd, int flags)
{
 struct audit_context *context = current->audit_context;
 context->mmap.fd = fd;
 context->mmap.flags = flags;
 context->type = AUDIT_MMAP;
}

static void audit_log_task(struct audit_buffer *ab)
{
 kuid_t auid, uid;
 kgid_t gid;
 unsigned int sessionid;
 char comm[sizeof(current->comm)];

 auid = audit_get_loginuid(current);
 sessionid = audit_get_sessionid(current);
 current_uid_gid(&uid, &gid);

 audit_log_format(ab, "auid=%u uid=%u gid=%u ses=%u",
    from_kuid(&init_user_ns, auid),
    from_kuid(&init_user_ns, uid),
    from_kgid(&init_user_ns, gid),
    sessionid);
 audit_log_task_context(ab);
 audit_log_format(ab, " pid=%d comm=", task_pid_nr(current));
 audit_log_untrustedstring(ab, get_task_comm(comm, current));
 audit_log_d_path_exe(ab, current->mm);
}
void audit_core_dumps(long signr)
{
 struct audit_buffer *ab;

 if (!audit_enabled)
  return;

 if (signr == SIGQUIT)
  return;

 ab = audit_log_start(NULL, GFP_KERNEL, AUDIT_ANOM_ABEND);
 if (unlikely(!ab))
  return;
 audit_log_task(ab);
 audit_log_format(ab, " sig=%ld", signr);
 audit_log_end(ab);
}

void __audit_seccomp(unsigned long syscall, long signr, int code)
{
 struct audit_buffer *ab;

 ab = audit_log_start(NULL, GFP_KERNEL, AUDIT_SECCOMP);
 if (unlikely(!ab))
  return;
 audit_log_task(ab);
 audit_log_format(ab, " sig=%ld arch=%x syscall=%ld compat=%d ip=0x%lx code=0x%x",
    signr, syscall_get_arch(), syscall,
    in_compat_syscall(), KSTK_EIP(current), code);
 audit_log_end(ab);
}

struct list_head *audit_killed_trees(void)
{
 struct audit_context *ctx = current->audit_context;
 if (likely(!ctx || !ctx->in_syscall))
  return NULL;
 return &ctx->killed_trees;
}

struct audit_tree;
struct audit_chunk;

struct audit_tree {
 atomic_t count;
 int goner;
 struct audit_chunk *root;
 struct list_head chunks;
 struct list_head rules;
 struct list_head list;
 struct list_head same_root;
 struct rcu_head head;
 char pathname[];
};

struct audit_chunk {
 struct list_head hash;
 struct fsnotify_mark mark;
 struct list_head trees;
 int dead;
 int count;
 atomic_long_t refs;
 struct rcu_head head;
 struct node {
  struct list_head list;
  struct audit_tree *owner;
  unsigned index;
 } owners[];
};

static LIST_HEAD(tree_list);
static LIST_HEAD(prune_list);
static struct task_struct *prune_thread;
static struct fsnotify_group *audit_tree_group;

static struct audit_tree *alloc_tree(const char *s)
{
 struct audit_tree *tree;

 tree = kmalloc(sizeof(struct audit_tree) + strlen(s) + 1, GFP_KERNEL);
 if (tree) {
  atomic_set(&tree->count, 1);
  tree->goner = 0;
  INIT_LIST_HEAD(&tree->chunks);
  INIT_LIST_HEAD(&tree->rules);
  INIT_LIST_HEAD(&tree->list);
  INIT_LIST_HEAD(&tree->same_root);
  tree->root = NULL;
  strcpy(tree->pathname, s);
 }
 return tree;
}

static inline void get_tree(struct audit_tree *tree)
{
 atomic_inc(&tree->count);
}

static inline void put_tree(struct audit_tree *tree)
{
 if (atomic_dec_and_test(&tree->count))
  kfree_rcu(tree, head);
}


const char *audit_tree_path(struct audit_tree *tree)
{
 return tree->pathname;
}

static void free_chunk(struct audit_chunk *chunk)
{
 int i;

 for (i = 0; i < chunk->count; i++) {
  if (chunk->owners[i].owner)
   put_tree(chunk->owners[i].owner);
 }
 kfree(chunk);
}

void audit_put_chunk(struct audit_chunk *chunk)
{
 if (atomic_long_dec_and_test(&chunk->refs))
  free_chunk(chunk);
}

static void __put_chunk(struct rcu_head *rcu)
{
 struct audit_chunk *chunk = container_of(rcu, struct audit_chunk, head);
 audit_put_chunk(chunk);
}

static void audit_tree_destroy_watch(struct fsnotify_mark *entry)
{
 struct audit_chunk *chunk = container_of(entry, struct audit_chunk, mark);
 call_rcu(&chunk->head, __put_chunk);
}

static struct audit_chunk *alloc_chunk(int count)
{
 struct audit_chunk *chunk;
 size_t size;
 int i;

 size = offsetof(struct audit_chunk, owners) + count * sizeof(struct node);
 chunk = kzalloc(size, GFP_KERNEL);
 if (!chunk)
  return NULL;

 INIT_LIST_HEAD(&chunk->hash);
 INIT_LIST_HEAD(&chunk->trees);
 chunk->count = count;
 atomic_long_set(&chunk->refs, 1);
 for (i = 0; i < count; i++) {
  INIT_LIST_HEAD(&chunk->owners[i].list);
  chunk->owners[i].index = i;
 }
 fsnotify_init_mark(&chunk->mark, audit_tree_destroy_watch);
 chunk->mark.mask = FS_IN_IGNORED;
 return chunk;
}

enum {HASH_SIZE = 128};
static struct list_head chunk_hash_heads[HASH_SIZE];
static __cacheline_aligned_in_smp DEFINE_SPINLOCK(hash_lock);

static inline struct list_head *chunk_hash(const struct inode *inode)
{
 unsigned long n = (unsigned long)inode / L1_CACHE_BYTES;
 return chunk_hash_heads + n % HASH_SIZE;
}


static void insert_hash(struct audit_chunk *chunk)
{
 struct fsnotify_mark *entry = &chunk->mark;
 struct list_head *list;

 if (!entry->inode)
  return;
 list = chunk_hash(entry->inode);
 list_add_rcu(&chunk->hash, list);
}


struct audit_chunk *audit_tree_lookup(const struct inode *inode)
{
 struct list_head *list = chunk_hash(inode);
 struct audit_chunk *p;

 list_for_each_entry_rcu(p, list, hash) {

  if (p->mark.inode == inode) {
   atomic_long_inc(&p->refs);
   return p;
  }
 }
 return NULL;
}

bool audit_tree_match(struct audit_chunk *chunk, struct audit_tree *tree)
{
 int n;
 for (n = 0; n < chunk->count; n++)
  if (chunk->owners[n].owner == tree)
   return true;
 return false;
}



static struct audit_chunk *find_chunk(struct node *p)
{
 int index = p->index & ~(1U<<31);
 p -= index;
 return container_of(p, struct audit_chunk, owners[0]);
}

static void untag_chunk(struct node *p)
{
 struct audit_chunk *chunk = find_chunk(p);
 struct fsnotify_mark *entry = &chunk->mark;
 struct audit_chunk *new = NULL;
 struct audit_tree *owner;
 int size = chunk->count - 1;
 int i, j;

 fsnotify_get_mark(entry);

 spin_unlock(&hash_lock);

 if (size)
  new = alloc_chunk(size);

 spin_lock(&entry->lock);
 if (chunk->dead || !entry->inode) {
  spin_unlock(&entry->lock);
  if (new)
   free_chunk(new);
  goto out;
 }

 owner = p->owner;

 if (!size) {
  chunk->dead = 1;
  spin_lock(&hash_lock);
  list_del_init(&chunk->trees);
  if (owner->root == chunk)
   owner->root = NULL;
  list_del_init(&p->list);
  list_del_rcu(&chunk->hash);
  spin_unlock(&hash_lock);
  spin_unlock(&entry->lock);
  fsnotify_destroy_mark(entry, audit_tree_group);
  goto out;
 }

 if (!new)
  goto Fallback;

 fsnotify_duplicate_mark(&new->mark, entry);
 if (fsnotify_add_mark(&new->mark, new->mark.group, new->mark.inode, NULL, 1)) {
  fsnotify_put_mark(&new->mark);
  goto Fallback;
 }

 chunk->dead = 1;
 spin_lock(&hash_lock);
 list_replace_init(&chunk->trees, &new->trees);
 if (owner->root == chunk) {
  list_del_init(&owner->same_root);
  owner->root = NULL;
 }

 for (i = j = 0; j <= size; i++, j++) {
  struct audit_tree *s;
  if (&chunk->owners[j] == p) {
   list_del_init(&p->list);
   i--;
   continue;
  }
  s = chunk->owners[j].owner;
  new->owners[i].owner = s;
  new->owners[i].index = chunk->owners[j].index - j + i;
  if (!s)
   continue;
  get_tree(s);
  list_replace_init(&chunk->owners[j].list, &new->owners[i].list);
 }

 list_replace_rcu(&chunk->hash, &new->hash);
 list_for_each_entry(owner, &new->trees, same_root)
  owner->root = new;
 spin_unlock(&hash_lock);
 spin_unlock(&entry->lock);
 fsnotify_destroy_mark(entry, audit_tree_group);
 fsnotify_put_mark(&new->mark);
 goto out;

Fallback:

 spin_lock(&hash_lock);
 if (owner->root == chunk) {
  list_del_init(&owner->same_root);
  owner->root = NULL;
 }
 list_del_init(&p->list);
 p->owner = NULL;
 put_tree(owner);
 spin_unlock(&hash_lock);
 spin_unlock(&entry->lock);
out:
 fsnotify_put_mark(entry);
 spin_lock(&hash_lock);
}

static int create_chunk(struct inode *inode, struct audit_tree *tree)
{
 struct fsnotify_mark *entry;
 struct audit_chunk *chunk = alloc_chunk(1);
 if (!chunk)
  return -ENOMEM;

 entry = &chunk->mark;
 if (fsnotify_add_mark(entry, audit_tree_group, inode, NULL, 0)) {
  fsnotify_put_mark(entry);
  return -ENOSPC;
 }

 spin_lock(&entry->lock);
 spin_lock(&hash_lock);
 if (tree->goner) {
  spin_unlock(&hash_lock);
  chunk->dead = 1;
  spin_unlock(&entry->lock);
  fsnotify_destroy_mark(entry, audit_tree_group);
  fsnotify_put_mark(entry);
  return 0;
 }
 chunk->owners[0].index = (1U << 31);
 chunk->owners[0].owner = tree;
 get_tree(tree);
 list_add(&chunk->owners[0].list, &tree->chunks);
 if (!tree->root) {
  tree->root = chunk;
  list_add(&tree->same_root, &chunk->trees);
 }
 insert_hash(chunk);
 spin_unlock(&hash_lock);
 spin_unlock(&entry->lock);
 fsnotify_put_mark(entry);
 return 0;
}


static int tag_chunk(struct inode *inode, struct audit_tree *tree)
{
 struct fsnotify_mark *old_entry, *chunk_entry;
 struct audit_tree *owner;
 struct audit_chunk *chunk, *old;
 struct node *p;
 int n;

 old_entry = fsnotify_find_inode_mark(audit_tree_group, inode);
 if (!old_entry)
  return create_chunk(inode, tree);

 old = container_of(old_entry, struct audit_chunk, mark);


 spin_lock(&hash_lock);
 for (n = 0; n < old->count; n++) {
  if (old->owners[n].owner == tree) {
   spin_unlock(&hash_lock);
   fsnotify_put_mark(old_entry);
   return 0;
  }
 }
 spin_unlock(&hash_lock);

 chunk = alloc_chunk(old->count + 1);
 if (!chunk) {
  fsnotify_put_mark(old_entry);
  return -ENOMEM;
 }

 chunk_entry = &chunk->mark;

 spin_lock(&old_entry->lock);
 if (!old_entry->inode) {

  spin_unlock(&old_entry->lock);
  fsnotify_put_mark(old_entry);
  free_chunk(chunk);
  return -ENOENT;
 }

 fsnotify_duplicate_mark(chunk_entry, old_entry);
 if (fsnotify_add_mark(chunk_entry, chunk_entry->group, chunk_entry->inode, NULL, 1)) {
  spin_unlock(&old_entry->lock);
  fsnotify_put_mark(chunk_entry);
  fsnotify_put_mark(old_entry);
  return -ENOSPC;
 }


 spin_lock(&chunk_entry->lock);
 spin_lock(&hash_lock);


 if (tree->goner) {
  spin_unlock(&hash_lock);
  chunk->dead = 1;
  spin_unlock(&chunk_entry->lock);
  spin_unlock(&old_entry->lock);

  fsnotify_destroy_mark(chunk_entry, audit_tree_group);

  fsnotify_put_mark(chunk_entry);
  fsnotify_put_mark(old_entry);
  return 0;
 }
 list_replace_init(&old->trees, &chunk->trees);
 for (n = 0, p = chunk->owners; n < old->count; n++, p++) {
  struct audit_tree *s = old->owners[n].owner;
  p->owner = s;
  p->index = old->owners[n].index;
  if (!s)
   continue;
  get_tree(s);
  list_replace_init(&old->owners[n].list, &p->list);
 }
 p->index = (chunk->count - 1) | (1U<<31);
 p->owner = tree;
 get_tree(tree);
 list_add(&p->list, &tree->chunks);
 list_replace_rcu(&old->hash, &chunk->hash);
 list_for_each_entry(owner, &chunk->trees, same_root)
  owner->root = chunk;
 old->dead = 1;
 if (!tree->root) {
  tree->root = chunk;
  list_add(&tree->same_root, &chunk->trees);
 }
 spin_unlock(&hash_lock);
 spin_unlock(&chunk_entry->lock);
 spin_unlock(&old_entry->lock);
 fsnotify_destroy_mark(old_entry, audit_tree_group);
 fsnotify_put_mark(chunk_entry);
 fsnotify_put_mark(old_entry);
 return 0;
}

static void audit_tree_log_remove_rule(struct audit_krule *rule)
{
 struct audit_buffer *ab;

 ab = audit_log_start(NULL, GFP_KERNEL, AUDIT_CONFIG_CHANGE);
 if (unlikely(!ab))
  return;
 audit_log_format(ab, "op=");
 audit_log_string(ab, "remove_rule");
 audit_log_format(ab, " dir=");
 audit_log_untrustedstring(ab, rule->tree->pathname);
 audit_log_key(ab, rule->filterkey);
 audit_log_format(ab, " list=%d res=1", rule->listnr);
 audit_log_end(ab);
}

static void kill_rules(struct audit_tree *tree)
{
 struct audit_krule *rule, *next;
 struct audit_entry *entry;

 list_for_each_entry_safe(rule, next, &tree->rules, rlist) {
  entry = container_of(rule, struct audit_entry, rule);

  list_del_init(&rule->rlist);
  if (rule->tree) {

   audit_tree_log_remove_rule(rule);
   if (entry->rule.exe)
    audit_remove_mark(entry->rule.exe);
   rule->tree = NULL;
   list_del_rcu(&entry->list);
   list_del(&entry->rule.list);
   call_rcu(&entry->rcu, audit_free_rule_rcu);
  }
 }
}




static void prune_one(struct audit_tree *victim)
{
 spin_lock(&hash_lock);
 while (!list_empty(&victim->chunks)) {
  struct node *p;

  p = list_entry(victim->chunks.next, struct node, list);

  untag_chunk(p);
 }
 spin_unlock(&hash_lock);
 put_tree(victim);
}



static void trim_marked(struct audit_tree *tree)
{
 struct list_head *p, *q;
 spin_lock(&hash_lock);
 if (tree->goner) {
  spin_unlock(&hash_lock);
  return;
 }

 for (p = tree->chunks.next; p != &tree->chunks; p = q) {
  struct node *node = list_entry(p, struct node, list);
  q = p->next;
  if (node->index & (1U<<31)) {
   list_del_init(p);
   list_add(p, &tree->chunks);
  }
 }

 while (!list_empty(&tree->chunks)) {
  struct node *node;

  node = list_entry(tree->chunks.next, struct node, list);


  if (!(node->index & (1U<<31)))
   break;

  untag_chunk(node);
 }
 if (!tree->root && !tree->goner) {
  tree->goner = 1;
  spin_unlock(&hash_lock);
  mutex_lock(&audit_filter_mutex);
  kill_rules(tree);
  list_del_init(&tree->list);
  mutex_unlock(&audit_filter_mutex);
  prune_one(tree);
 } else {
  spin_unlock(&hash_lock);
 }
}

static void audit_schedule_prune(void);


int audit_remove_tree_rule(struct audit_krule *rule)
{
 struct audit_tree *tree;
 tree = rule->tree;
 if (tree) {
  spin_lock(&hash_lock);
  list_del_init(&rule->rlist);
  if (list_empty(&tree->rules) && !tree->goner) {
   tree->root = NULL;
   list_del_init(&tree->same_root);
   tree->goner = 1;
   list_move(&tree->list, &prune_list);
   rule->tree = NULL;
   spin_unlock(&hash_lock);
   audit_schedule_prune();
   return 1;
  }
  rule->tree = NULL;
  spin_unlock(&hash_lock);
  return 1;
 }
 return 0;
}

static int compare_root(struct vfsmount *mnt, void *arg)
{
 return d_backing_inode(mnt->mnt_root) == arg;
}

void audit_trim_trees(void)
{
 struct list_head cursor;

 mutex_lock(&audit_filter_mutex);
 list_add(&cursor, &tree_list);
 while (cursor.next != &tree_list) {
  struct audit_tree *tree;
  struct path path;
  struct vfsmount *root_mnt;
  struct node *node;
  int err;

  tree = container_of(cursor.next, struct audit_tree, list);
  get_tree(tree);
  list_del(&cursor);
  list_add(&cursor, &tree->list);
  mutex_unlock(&audit_filter_mutex);

  err = kern_path(tree->pathname, 0, &path);
  if (err)
   goto skip_it;

  root_mnt = collect_mounts(&path);
  path_put(&path);
  if (IS_ERR(root_mnt))
   goto skip_it;

  spin_lock(&hash_lock);
  list_for_each_entry(node, &tree->chunks, list) {
   struct audit_chunk *chunk = find_chunk(node);

   struct inode *inode = chunk->mark.inode;
   node->index |= 1U<<31;
   if (iterate_mounts(compare_root, inode, root_mnt))
    node->index &= ~(1U<<31);
  }
  spin_unlock(&hash_lock);
  trim_marked(tree);
  drop_collected_mounts(root_mnt);
skip_it:
  put_tree(tree);
  mutex_lock(&audit_filter_mutex);
 }
 list_del(&cursor);
 mutex_unlock(&audit_filter_mutex);
}

int audit_make_tree(struct audit_krule *rule, char *pathname, u32 op)
{

 if (pathname[0] != '/' ||
     rule->listnr != AUDIT_FILTER_EXIT ||
     op != Audit_equal ||
     rule->inode_f || rule->watch || rule->tree)
  return -EINVAL;
 rule->tree = alloc_tree(pathname);
 if (!rule->tree)
  return -ENOMEM;
 return 0;
}

void audit_put_tree(struct audit_tree *tree)
{
 put_tree(tree);
}

static int tag_mount(struct vfsmount *mnt, void *arg)
{
 return tag_chunk(d_backing_inode(mnt->mnt_root), arg);
}





static int prune_tree_thread(void *unused)
{
 for (;;) {
  if (list_empty(&prune_list)) {
   set_current_state(TASK_INTERRUPTIBLE);
   schedule();
  }

  mutex_lock(&audit_cmd_mutex);
  mutex_lock(&audit_filter_mutex);

  while (!list_empty(&prune_list)) {
   struct audit_tree *victim;

   victim = list_entry(prune_list.next,
     struct audit_tree, list);
   list_del_init(&victim->list);

   mutex_unlock(&audit_filter_mutex);

   prune_one(victim);

   mutex_lock(&audit_filter_mutex);
  }

  mutex_unlock(&audit_filter_mutex);
  mutex_unlock(&audit_cmd_mutex);
 }
 return 0;
}

static int audit_launch_prune(void)
{
 if (prune_thread)
  return 0;
 prune_thread = kthread_run(prune_tree_thread, NULL,
    "audit_prune_tree");
 if (IS_ERR(prune_thread)) {
  pr_err("cannot start thread audit_prune_tree");
  prune_thread = NULL;
  return -ENOMEM;
 }
 return 0;
}


int audit_add_tree_rule(struct audit_krule *rule)
{
 struct audit_tree *seed = rule->tree, *tree;
 struct path path;
 struct vfsmount *mnt;
 int err;

 rule->tree = NULL;
 list_for_each_entry(tree, &tree_list, list) {
  if (!strcmp(seed->pathname, tree->pathname)) {
   put_tree(seed);
   rule->tree = tree;
   list_add(&rule->rlist, &tree->rules);
   return 0;
  }
 }
 tree = seed;
 list_add(&tree->list, &tree_list);
 list_add(&rule->rlist, &tree->rules);

 mutex_unlock(&audit_filter_mutex);

 if (unlikely(!prune_thread)) {
  err = audit_launch_prune();
  if (err)
   goto Err;
 }

 err = kern_path(tree->pathname, 0, &path);
 if (err)
  goto Err;
 mnt = collect_mounts(&path);
 path_put(&path);
 if (IS_ERR(mnt)) {
  err = PTR_ERR(mnt);
  goto Err;
 }

 get_tree(tree);
 err = iterate_mounts(tag_mount, tree, mnt);
 drop_collected_mounts(mnt);

 if (!err) {
  struct node *node;
  spin_lock(&hash_lock);
  list_for_each_entry(node, &tree->chunks, list)
   node->index &= ~(1U<<31);
  spin_unlock(&hash_lock);
 } else {
  trim_marked(tree);
  goto Err;
 }

 mutex_lock(&audit_filter_mutex);
 if (list_empty(&rule->rlist)) {
  put_tree(tree);
  return -ENOENT;
 }
 rule->tree = tree;
 put_tree(tree);

 return 0;
Err:
 mutex_lock(&audit_filter_mutex);
 list_del_init(&tree->list);
 list_del_init(&tree->rules);
 put_tree(tree);
 return err;
}

int audit_tag_tree(char *old, char *new)
{
 struct list_head cursor, barrier;
 int failed = 0;
 struct path path1, path2;
 struct vfsmount *tagged;
 int err;

 err = kern_path(new, 0, &path2);
 if (err)
  return err;
 tagged = collect_mounts(&path2);
 path_put(&path2);
 if (IS_ERR(tagged))
  return PTR_ERR(tagged);

 err = kern_path(old, 0, &path1);
 if (err) {
  drop_collected_mounts(tagged);
  return err;
 }

 mutex_lock(&audit_filter_mutex);
 list_add(&barrier, &tree_list);
 list_add(&cursor, &barrier);

 while (cursor.next != &tree_list) {
  struct audit_tree *tree;
  int good_one = 0;

  tree = container_of(cursor.next, struct audit_tree, list);
  get_tree(tree);
  list_del(&cursor);
  list_add(&cursor, &tree->list);
  mutex_unlock(&audit_filter_mutex);

  err = kern_path(tree->pathname, 0, &path2);
  if (!err) {
   good_one = path_is_under(&path1, &path2);
   path_put(&path2);
  }

  if (!good_one) {
   put_tree(tree);
   mutex_lock(&audit_filter_mutex);
   continue;
  }

  failed = iterate_mounts(tag_mount, tree, tagged);
  if (failed) {
   put_tree(tree);
   mutex_lock(&audit_filter_mutex);
   break;
  }

  mutex_lock(&audit_filter_mutex);
  spin_lock(&hash_lock);
  if (!tree->goner) {
   list_del(&tree->list);
   list_add(&tree->list, &tree_list);
  }
  spin_unlock(&hash_lock);
  put_tree(tree);
 }

 while (barrier.prev != &tree_list) {
  struct audit_tree *tree;

  tree = container_of(barrier.prev, struct audit_tree, list);
  get_tree(tree);
  list_del(&tree->list);
  list_add(&tree->list, &barrier);
  mutex_unlock(&audit_filter_mutex);

  if (!failed) {
   struct node *node;
   spin_lock(&hash_lock);
   list_for_each_entry(node, &tree->chunks, list)
    node->index &= ~(1U<<31);
   spin_unlock(&hash_lock);
  } else {
   trim_marked(tree);
  }

  put_tree(tree);
  mutex_lock(&audit_filter_mutex);
 }
 list_del(&barrier);
 list_del(&cursor);
 mutex_unlock(&audit_filter_mutex);
 path_put(&path1);
 drop_collected_mounts(tagged);
 return failed;
}


static void audit_schedule_prune(void)
{
 wake_up_process(prune_thread);
}





void audit_kill_trees(struct list_head *list)
{
 mutex_lock(&audit_cmd_mutex);
 mutex_lock(&audit_filter_mutex);

 while (!list_empty(list)) {
  struct audit_tree *victim;

  victim = list_entry(list->next, struct audit_tree, list);
  kill_rules(victim);
  list_del_init(&victim->list);

  mutex_unlock(&audit_filter_mutex);

  prune_one(victim);

  mutex_lock(&audit_filter_mutex);
 }

 mutex_unlock(&audit_filter_mutex);
 mutex_unlock(&audit_cmd_mutex);
}





static void evict_chunk(struct audit_chunk *chunk)
{
 struct audit_tree *owner;
 struct list_head *postponed = audit_killed_trees();
 int need_prune = 0;
 int n;

 if (chunk->dead)
  return;

 chunk->dead = 1;
 mutex_lock(&audit_filter_mutex);
 spin_lock(&hash_lock);
 while (!list_empty(&chunk->trees)) {
  owner = list_entry(chunk->trees.next,
       struct audit_tree, same_root);
  owner->goner = 1;
  owner->root = NULL;
  list_del_init(&owner->same_root);
  spin_unlock(&hash_lock);
  if (!postponed) {
   kill_rules(owner);
   list_move(&owner->list, &prune_list);
   need_prune = 1;
  } else {
   list_move(&owner->list, postponed);
  }
  spin_lock(&hash_lock);
 }
 list_del_rcu(&chunk->hash);
 for (n = 0; n < chunk->count; n++)
  list_del_init(&chunk->owners[n].list);
 spin_unlock(&hash_lock);
 mutex_unlock(&audit_filter_mutex);
 if (need_prune)
  audit_schedule_prune();
}

static int audit_tree_handle_event(struct fsnotify_group *group,
       struct inode *to_tell,
       struct fsnotify_mark *inode_mark,
       struct fsnotify_mark *vfsmount_mark,
       u32 mask, void *data, int data_type,
       const unsigned char *file_name, u32 cookie)
{
 return 0;
}

static void audit_tree_freeing_mark(struct fsnotify_mark *entry, struct fsnotify_group *group)
{
 struct audit_chunk *chunk = container_of(entry, struct audit_chunk, mark);

 evict_chunk(chunk);





 BUG_ON(atomic_read(&entry->refcnt) < 1);
}

static const struct fsnotify_ops audit_tree_ops = {
 .handle_event = audit_tree_handle_event,
 .freeing_mark = audit_tree_freeing_mark,
};

static int __init audit_tree_init(void)
{
 int i;

 audit_tree_group = fsnotify_alloc_group(&audit_tree_ops);
 if (IS_ERR(audit_tree_group))
  audit_panic("cannot initialize fsnotify group for rectree watches");

 for (i = 0; i < HASH_SIZE; i++)
  INIT_LIST_HEAD(&chunk_hash_heads[i]);

 return 0;
}
__initcall(audit_tree_init);
struct audit_watch {
 atomic_t count;
 dev_t dev;
 char *path;
 unsigned long ino;
 struct audit_parent *parent;
 struct list_head wlist;
 struct list_head rules;
};

struct audit_parent {
 struct list_head watches;
 struct fsnotify_mark mark;
};


static struct fsnotify_group *audit_watch_group;


   FS_MOVE_SELF | FS_EVENT_ON_CHILD)

static void audit_free_parent(struct audit_parent *parent)
{
 WARN_ON(!list_empty(&parent->watches));
 kfree(parent);
}

static void audit_watch_free_mark(struct fsnotify_mark *entry)
{
 struct audit_parent *parent;

 parent = container_of(entry, struct audit_parent, mark);
 audit_free_parent(parent);
}

static void audit_get_parent(struct audit_parent *parent)
{
 if (likely(parent))
  fsnotify_get_mark(&parent->mark);
}

static void audit_put_parent(struct audit_parent *parent)
{
 if (likely(parent))
  fsnotify_put_mark(&parent->mark);
}





static inline struct audit_parent *audit_find_parent(struct inode *inode)
{
 struct audit_parent *parent = NULL;
 struct fsnotify_mark *entry;

 entry = fsnotify_find_inode_mark(audit_watch_group, inode);
 if (entry)
  parent = container_of(entry, struct audit_parent, mark);

 return parent;
}

void audit_get_watch(struct audit_watch *watch)
{
 atomic_inc(&watch->count);
}

void audit_put_watch(struct audit_watch *watch)
{
 if (atomic_dec_and_test(&watch->count)) {
  WARN_ON(watch->parent);
  WARN_ON(!list_empty(&watch->rules));
  kfree(watch->path);
  kfree(watch);
 }
}

static void audit_remove_watch(struct audit_watch *watch)
{
 list_del(&watch->wlist);
 audit_put_parent(watch->parent);
 watch->parent = NULL;
 audit_put_watch(watch);
}

char *audit_watch_path(struct audit_watch *watch)
{
 return watch->path;
}

int audit_watch_compare(struct audit_watch *watch, unsigned long ino, dev_t dev)
{
 return (watch->ino != AUDIT_INO_UNSET) &&
  (watch->ino == ino) &&
  (watch->dev == dev);
}


static struct audit_parent *audit_init_parent(struct path *path)
{
 struct inode *inode = d_backing_inode(path->dentry);
 struct audit_parent *parent;
 int ret;

 parent = kzalloc(sizeof(*parent), GFP_KERNEL);
 if (unlikely(!parent))
  return ERR_PTR(-ENOMEM);

 INIT_LIST_HEAD(&parent->watches);

 fsnotify_init_mark(&parent->mark, audit_watch_free_mark);
 parent->mark.mask = AUDIT_FS_WATCH;
 ret = fsnotify_add_mark(&parent->mark, audit_watch_group, inode, NULL, 0);
 if (ret < 0) {
  audit_free_parent(parent);
  return ERR_PTR(ret);
 }

 return parent;
}


static struct audit_watch *audit_init_watch(char *path)
{
 struct audit_watch *watch;

 watch = kzalloc(sizeof(*watch), GFP_KERNEL);
 if (unlikely(!watch))
  return ERR_PTR(-ENOMEM);

 INIT_LIST_HEAD(&watch->rules);
 atomic_set(&watch->count, 1);
 watch->path = path;
 watch->dev = AUDIT_DEV_UNSET;
 watch->ino = AUDIT_INO_UNSET;

 return watch;
}


int audit_to_watch(struct audit_krule *krule, char *path, int len, u32 op)
{
 struct audit_watch *watch;

 if (!audit_watch_group)
  return -EOPNOTSUPP;

 if (path[0] != '/' || path[len-1] == '/' ||
     krule->listnr != AUDIT_FILTER_EXIT ||
     op != Audit_equal ||
     krule->inode_f || krule->watch || krule->tree)
  return -EINVAL;

 watch = audit_init_watch(path);
 if (IS_ERR(watch))
  return PTR_ERR(watch);

 krule->watch = watch;

 return 0;
}



static struct audit_watch *audit_dupe_watch(struct audit_watch *old)
{
 char *path;
 struct audit_watch *new;

 path = kstrdup(old->path, GFP_KERNEL);
 if (unlikely(!path))
  return ERR_PTR(-ENOMEM);

 new = audit_init_watch(path);
 if (IS_ERR(new)) {
  kfree(path);
  goto out;
 }

 new->dev = old->dev;
 new->ino = old->ino;
 audit_get_parent(old->parent);
 new->parent = old->parent;

out:
 return new;
}

static void audit_watch_log_rule_change(struct audit_krule *r, struct audit_watch *w, char *op)
{
 if (audit_enabled) {
  struct audit_buffer *ab;
  ab = audit_log_start(NULL, GFP_NOFS, AUDIT_CONFIG_CHANGE);
  if (unlikely(!ab))
   return;
  audit_log_format(ab, "auid=%u ses=%u op=",
     from_kuid(&init_user_ns, audit_get_loginuid(current)),
     audit_get_sessionid(current));
  audit_log_string(ab, op);
  audit_log_format(ab, " path=");
  audit_log_untrustedstring(ab, w->path);
  audit_log_key(ab, r->filterkey);
  audit_log_format(ab, " list=%d res=1", r->listnr);
  audit_log_end(ab);
 }
}


static void audit_update_watch(struct audit_parent *parent,
          const char *dname, dev_t dev,
          unsigned long ino, unsigned invalidating)
{
 struct audit_watch *owatch, *nwatch, *nextw;
 struct audit_krule *r, *nextr;
 struct audit_entry *oentry, *nentry;

 mutex_lock(&audit_filter_mutex);


 list_for_each_entry_safe(owatch, nextw, &parent->watches, wlist) {
  if (audit_compare_dname_path(dname, owatch->path,
          AUDIT_NAME_FULL))
   continue;



  if (invalidating && !audit_dummy_context())
   audit_filter_inodes(current, current->audit_context);



  nwatch = audit_dupe_watch(owatch);
  if (IS_ERR(nwatch)) {
   mutex_unlock(&audit_filter_mutex);
   audit_panic("error updating watch, skipping");
   return;
  }
  nwatch->dev = dev;
  nwatch->ino = ino;

  list_for_each_entry_safe(r, nextr, &owatch->rules, rlist) {

   oentry = container_of(r, struct audit_entry, rule);
   list_del(&oentry->rule.rlist);
   list_del_rcu(&oentry->list);

   nentry = audit_dupe_rule(&oentry->rule);
   if (IS_ERR(nentry)) {
    list_del(&oentry->rule.list);
    audit_panic("error updating watch, removing");
   } else {
    int h = audit_hash_ino((u32)ino);






    audit_put_watch(nentry->rule.watch);
    audit_get_watch(nwatch);
    nentry->rule.watch = nwatch;
    list_add(&nentry->rule.rlist, &nwatch->rules);
    list_add_rcu(&nentry->list, &audit_inode_hash[h]);
    list_replace(&oentry->rule.list,
          &nentry->rule.list);
   }
   if (oentry->rule.exe)
    audit_remove_mark(oentry->rule.exe);

   audit_watch_log_rule_change(r, owatch, "updated_rules");

   call_rcu(&oentry->rcu, audit_free_rule_rcu);
  }

  audit_remove_watch(owatch);
  goto add_watch_to_parent;
 }
 mutex_unlock(&audit_filter_mutex);
 return;

add_watch_to_parent:
 list_add(&nwatch->wlist, &parent->watches);
 mutex_unlock(&audit_filter_mutex);
 return;
}


static void audit_remove_parent_watches(struct audit_parent *parent)
{
 struct audit_watch *w, *nextw;
 struct audit_krule *r, *nextr;
 struct audit_entry *e;

 mutex_lock(&audit_filter_mutex);
 list_for_each_entry_safe(w, nextw, &parent->watches, wlist) {
  list_for_each_entry_safe(r, nextr, &w->rules, rlist) {
   e = container_of(r, struct audit_entry, rule);
   audit_watch_log_rule_change(r, w, "remove_rule");
   if (e->rule.exe)
    audit_remove_mark(e->rule.exe);
   list_del(&r->rlist);
   list_del(&r->list);
   list_del_rcu(&e->list);
   call_rcu(&e->rcu, audit_free_rule_rcu);
  }
  audit_remove_watch(w);
 }
 mutex_unlock(&audit_filter_mutex);

 fsnotify_destroy_mark(&parent->mark, audit_watch_group);
}


static int audit_get_nd(struct audit_watch *watch, struct path *parent)
{
 struct dentry *d = kern_path_locked(watch->path, parent);
 if (IS_ERR(d))
  return PTR_ERR(d);
 inode_unlock(d_backing_inode(parent->dentry));
 if (d_is_positive(d)) {

  watch->dev = d->d_sb->s_dev;
  watch->ino = d_backing_inode(d)->i_ino;
 }
 dput(d);
 return 0;
}



static void audit_add_to_parent(struct audit_krule *krule,
    struct audit_parent *parent)
{
 struct audit_watch *w, *watch = krule->watch;
 int watch_found = 0;

 BUG_ON(!mutex_is_locked(&audit_filter_mutex));

 list_for_each_entry(w, &parent->watches, wlist) {
  if (strcmp(watch->path, w->path))
   continue;

  watch_found = 1;


  audit_put_watch(watch);

  audit_get_watch(w);
  krule->watch = watch = w;

  audit_put_parent(parent);
  break;
 }

 if (!watch_found) {
  watch->parent = parent;

  audit_get_watch(watch);
  list_add(&watch->wlist, &parent->watches);
 }
 list_add(&krule->rlist, &watch->rules);
}



int audit_add_watch(struct audit_krule *krule, struct list_head **list)
{
 struct audit_watch *watch = krule->watch;
 struct audit_parent *parent;
 struct path parent_path;
 int h, ret = 0;

 mutex_unlock(&audit_filter_mutex);


 ret = audit_get_nd(watch, &parent_path);


 mutex_lock(&audit_filter_mutex);

 if (ret)
  return ret;


 parent = audit_find_parent(d_backing_inode(parent_path.dentry));
 if (!parent) {
  parent = audit_init_parent(&parent_path);
  if (IS_ERR(parent)) {
   ret = PTR_ERR(parent);
   goto error;
  }
 }

 audit_add_to_parent(krule, parent);

 h = audit_hash_ino((u32)watch->ino);
 *list = &audit_inode_hash[h];
error:
 path_put(&parent_path);
 return ret;
}

void audit_remove_watch_rule(struct audit_krule *krule)
{
 struct audit_watch *watch = krule->watch;
 struct audit_parent *parent = watch->parent;

 list_del(&krule->rlist);

 if (list_empty(&watch->rules)) {
  audit_remove_watch(watch);

  if (list_empty(&parent->watches)) {
   audit_get_parent(parent);
   fsnotify_destroy_mark(&parent->mark, audit_watch_group);
   audit_put_parent(parent);
  }
 }
}


static int audit_watch_handle_event(struct fsnotify_group *group,
        struct inode *to_tell,
        struct fsnotify_mark *inode_mark,
        struct fsnotify_mark *vfsmount_mark,
        u32 mask, void *data, int data_type,
        const unsigned char *dname, u32 cookie)
{
 struct inode *inode;
 struct audit_parent *parent;

 parent = container_of(inode_mark, struct audit_parent, mark);

 BUG_ON(group != audit_watch_group);

 switch (data_type) {
 case (FSNOTIFY_EVENT_PATH):
  inode = d_backing_inode(((struct path *)data)->dentry);
  break;
 case (FSNOTIFY_EVENT_INODE):
  inode = (struct inode *)data;
  break;
 default:
  BUG();
  inode = NULL;
  break;
 };

 if (mask & (FS_CREATE|FS_MOVED_TO) && inode)
  audit_update_watch(parent, dname, inode->i_sb->s_dev, inode->i_ino, 0);
 else if (mask & (FS_DELETE|FS_MOVED_FROM))
  audit_update_watch(parent, dname, AUDIT_DEV_UNSET, AUDIT_INO_UNSET, 1);
 else if (mask & (FS_DELETE_SELF|FS_UNMOUNT|FS_MOVE_SELF))
  audit_remove_parent_watches(parent);

 return 0;
}

static const struct fsnotify_ops audit_watch_fsnotify_ops = {
 .handle_event = audit_watch_handle_event,
};

static int __init audit_watch_init(void)
{
 audit_watch_group = fsnotify_alloc_group(&audit_watch_fsnotify_ops);
 if (IS_ERR(audit_watch_group)) {
  audit_watch_group = NULL;
  audit_panic("cannot create audit fsnotify group");
 }
 return 0;
}
device_initcall(audit_watch_init);

int audit_dupe_exe(struct audit_krule *new, struct audit_krule *old)
{
 struct audit_fsnotify_mark *audit_mark;
 char *pathname;

 pathname = kstrdup(audit_mark_path(old->exe), GFP_KERNEL);
 if (!pathname)
  return -ENOMEM;

 audit_mark = audit_alloc_mark(new, pathname, strlen(pathname));
 if (IS_ERR(audit_mark)) {
  kfree(pathname);
  return PTR_ERR(audit_mark);
 }
 new->exe = audit_mark;

 return 0;
}

int audit_exe_compare(struct task_struct *tsk, struct audit_fsnotify_mark *mark)
{
 struct file *exe_file;
 unsigned long ino;
 dev_t dev;

 rcu_read_lock();
 exe_file = rcu_dereference(tsk->mm->exe_file);
 ino = exe_file->f_inode->i_ino;
 dev = exe_file->f_inode->i_sb->s_dev;
 rcu_read_unlock();
 return audit_mark_compare(mark, ino, dev);
}







static DEFINE_MUTEX(probing_active);
unsigned long probe_irq_on(void)
{
 struct irq_desc *desc;
 unsigned long mask = 0;
 int i;




 async_synchronize_full();
 mutex_lock(&probing_active);




 for_each_irq_desc_reverse(i, desc) {
  raw_spin_lock_irq(&desc->lock);
  if (!desc->action && irq_settings_can_probe(desc)) {




   if (desc->irq_data.chip->irq_set_type)
    desc->irq_data.chip->irq_set_type(&desc->irq_data,
        IRQ_TYPE_PROBE);
   irq_startup(desc, false);
  }
  raw_spin_unlock_irq(&desc->lock);
 }


 msleep(20);






 for_each_irq_desc_reverse(i, desc) {
  raw_spin_lock_irq(&desc->lock);
  if (!desc->action && irq_settings_can_probe(desc)) {
   desc->istate |= IRQS_AUTODETECT | IRQS_WAITING;
   if (irq_startup(desc, false))
    desc->istate |= IRQS_PENDING;
  }
  raw_spin_unlock_irq(&desc->lock);
 }




 msleep(100);




 for_each_irq_desc(i, desc) {
  raw_spin_lock_irq(&desc->lock);

  if (desc->istate & IRQS_AUTODETECT) {

   if (!(desc->istate & IRQS_WAITING)) {
    desc->istate &= ~IRQS_AUTODETECT;
    irq_shutdown(desc);
   } else
    if (i < 32)
     mask |= 1 << i;
  }
  raw_spin_unlock_irq(&desc->lock);
 }

 return mask;
}
EXPORT_SYMBOL(probe_irq_on);
unsigned int probe_irq_mask(unsigned long val)
{
 unsigned int mask = 0;
 struct irq_desc *desc;
 int i;

 for_each_irq_desc(i, desc) {
  raw_spin_lock_irq(&desc->lock);
  if (desc->istate & IRQS_AUTODETECT) {
   if (i < 16 && !(desc->istate & IRQS_WAITING))
    mask |= 1 << i;

   desc->istate &= ~IRQS_AUTODETECT;
   irq_shutdown(desc);
  }
  raw_spin_unlock_irq(&desc->lock);
 }
 mutex_unlock(&probing_active);

 return mask & val;
}
EXPORT_SYMBOL(probe_irq_mask);
int probe_irq_off(unsigned long val)
{
 int i, irq_found = 0, nr_of_irqs = 0;
 struct irq_desc *desc;

 for_each_irq_desc(i, desc) {
  raw_spin_lock_irq(&desc->lock);

  if (desc->istate & IRQS_AUTODETECT) {
   if (!(desc->istate & IRQS_WAITING)) {
    if (!nr_of_irqs)
     irq_found = i;
    nr_of_irqs++;
   }
   desc->istate &= ~IRQS_AUTODETECT;
   irq_shutdown(desc);
  }
  raw_spin_unlock_irq(&desc->lock);
 }
 mutex_unlock(&probing_active);

 if (nr_of_irqs > 1)
  irq_found = -irq_found;

 return irq_found;
}
EXPORT_SYMBOL(probe_irq_off);


static suspend_state_t autosleep_state;
static struct workqueue_struct *autosleep_wq;






static DEFINE_MUTEX(autosleep_lock);
static struct wakeup_source *autosleep_ws;

static void try_to_suspend(struct work_struct *work)
{
 unsigned int initial_count, final_count;

 if (!pm_get_wakeup_count(&initial_count, true))
  goto out;

 mutex_lock(&autosleep_lock);

 if (!pm_save_wakeup_count(initial_count) ||
  system_state != SYSTEM_RUNNING) {
  mutex_unlock(&autosleep_lock);
  goto out;
 }

 if (autosleep_state == PM_SUSPEND_ON) {
  mutex_unlock(&autosleep_lock);
  return;
 }
 if (autosleep_state >= PM_SUSPEND_MAX)
  hibernate();
 else
  pm_suspend(autosleep_state);

 mutex_unlock(&autosleep_lock);

 if (!pm_get_wakeup_count(&final_count, false))
  goto out;





 if (final_count == initial_count)
  schedule_timeout_uninterruptible(HZ / 2);

 out:
 queue_up_suspend_work();
}

static DECLARE_WORK(suspend_work, try_to_suspend);

void queue_up_suspend_work(void)
{
 if (autosleep_state > PM_SUSPEND_ON)
  queue_work(autosleep_wq, &suspend_work);
}

suspend_state_t pm_autosleep_state(void)
{
 return autosleep_state;
}

int pm_autosleep_lock(void)
{
 return mutex_lock_interruptible(&autosleep_lock);
}

void pm_autosleep_unlock(void)
{
 mutex_unlock(&autosleep_lock);
}

int pm_autosleep_set_state(suspend_state_t state)
{

 if (state >= PM_SUSPEND_MAX)
  return -EINVAL;

 __pm_stay_awake(autosleep_ws);

 mutex_lock(&autosleep_lock);

 autosleep_state = state;

 __pm_relax(autosleep_ws);

 if (state > PM_SUSPEND_ON) {
  pm_wakep_autosleep_enabled(true);
  queue_up_suspend_work();
 } else {
  pm_wakep_autosleep_enabled(false);
 }

 mutex_unlock(&autosleep_lock);
 return 0;
}

int __init pm_autosleep_init(void)
{
 autosleep_ws = wakeup_source_register("autosleep");
 if (!autosleep_ws)
  return -ENOMEM;

 autosleep_wq = alloc_ordered_workqueue("autosleep", 0);
 if (autosleep_wq)
  return 0;

 wakeup_source_unregister(autosleep_ws);
 return -ENOMEM;
}

static void backtrace_test_normal(void)
{
 pr_info("Testing a backtrace from process context.\n");
 pr_info("The following trace is a kernel self test and not a bug!\n");

 dump_stack();
}

static DECLARE_COMPLETION(backtrace_work);

static void backtrace_test_irq_callback(unsigned long data)
{
 dump_stack();
 complete(&backtrace_work);
}

static DECLARE_TASKLET(backtrace_tasklet, &backtrace_test_irq_callback, 0);

static void backtrace_test_irq(void)
{
 pr_info("Testing a backtrace from irq context.\n");
 pr_info("The following trace is a kernel self test and not a bug!\n");

 init_completion(&backtrace_work);
 tasklet_schedule(&backtrace_tasklet);
 wait_for_completion(&backtrace_work);
}

static void backtrace_test_saved(void)
{
 struct stack_trace trace;
 unsigned long entries[8];

 pr_info("Testing a saved backtrace.\n");
 pr_info("The following trace is a kernel self test and not a bug!\n");

 trace.nr_entries = 0;
 trace.max_entries = ARRAY_SIZE(entries);
 trace.entries = entries;
 trace.skip = 0;

 save_stack_trace(&trace);
 print_stack_trace(&trace, 0);
}
static void backtrace_test_saved(void)
{
 pr_info("Saved backtrace test skipped.\n");
}

static int backtrace_regression_test(void)
{
 pr_info("====[ backtrace testing ]===========\n");

 backtrace_test_normal();
 backtrace_test_irq();
 backtrace_test_saved();

 pr_info("====[ end of backtrace testing ]====\n");
 return 0;
}

static void exitf(void)
{
}

module_init(backtrace_regression_test);
module_exit(exitf);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Arjan van de Ven <arjan@linux.intel.com>");








void foo(void)
{

 DEFINE(NR_PAGEFLAGS, __NR_PAGEFLAGS);
 DEFINE(MAX_NR_ZONES, __MAX_NR_ZONES);
 DEFINE(NR_CPUS_BITS, ilog2(CONFIG_NR_CPUS));
 DEFINE(SPINLOCK_SIZE, sizeof(spinlock_t));

}



char *_braille_console_setup(char **str, char **brl_options)
{
 if (!memcmp(*str, "brl,", 4)) {
  *brl_options = "";
  *str += 4;
 } else if (!memcmp(str, "brl=", 4)) {
  *brl_options = *str + 4;
  *str = strchr(*brl_options, ',');
  if (!*str)
   pr_err("need port name after brl=\n");
  else
   *((*str)++) = 0;
 } else
  return NULL;

 return *str;
}

int
_braille_register_console(struct console *console, struct console_cmdline *c)
{
 int rtn = 0;

 if (c->brl_options) {
  console->flags |= CON_BRL;
  rtn = braille_register_console(console, c->index, c->options,
            c->brl_options);
 }

 return rtn;
}

int
_braille_unregister_console(struct console *console)
{
 if (console->flags & CON_BRL)
  return braille_unregister_console(console);

 return 0;
}

struct callchain_cpus_entries {
 struct rcu_head rcu_head;
 struct perf_callchain_entry *cpu_entries[0];
};

int sysctl_perf_event_max_stack __read_mostly = PERF_MAX_STACK_DEPTH;
int sysctl_perf_event_max_contexts_per_stack __read_mostly = PERF_MAX_CONTEXTS_PER_STACK;

static inline size_t perf_callchain_entry__sizeof(void)
{
 return (sizeof(struct perf_callchain_entry) +
  sizeof(__u64) * (sysctl_perf_event_max_stack +
     sysctl_perf_event_max_contexts_per_stack));
}

static DEFINE_PER_CPU(int, callchain_recursion[PERF_NR_CONTEXTS]);
static atomic_t nr_callchain_events;
static DEFINE_MUTEX(callchain_mutex);
static struct callchain_cpus_entries *callchain_cpus_entries;


__weak void perf_callchain_kernel(struct perf_callchain_entry_ctx *entry,
      struct pt_regs *regs)
{
}

__weak void perf_callchain_user(struct perf_callchain_entry_ctx *entry,
    struct pt_regs *regs)
{
}

static void release_callchain_buffers_rcu(struct rcu_head *head)
{
 struct callchain_cpus_entries *entries;
 int cpu;

 entries = container_of(head, struct callchain_cpus_entries, rcu_head);

 for_each_possible_cpu(cpu)
  kfree(entries->cpu_entries[cpu]);

 kfree(entries);
}

static void release_callchain_buffers(void)
{
 struct callchain_cpus_entries *entries;

 entries = callchain_cpus_entries;
 RCU_INIT_POINTER(callchain_cpus_entries, NULL);
 call_rcu(&entries->rcu_head, release_callchain_buffers_rcu);
}

static int alloc_callchain_buffers(void)
{
 int cpu;
 int size;
 struct callchain_cpus_entries *entries;






 size = offsetof(struct callchain_cpus_entries, cpu_entries[nr_cpu_ids]);

 entries = kzalloc(size, GFP_KERNEL);
 if (!entries)
  return -ENOMEM;

 size = perf_callchain_entry__sizeof() * PERF_NR_CONTEXTS;

 for_each_possible_cpu(cpu) {
  entries->cpu_entries[cpu] = kmalloc_node(size, GFP_KERNEL,
        cpu_to_node(cpu));
  if (!entries->cpu_entries[cpu])
   goto fail;
 }

 rcu_assign_pointer(callchain_cpus_entries, entries);

 return 0;

fail:
 for_each_possible_cpu(cpu)
  kfree(entries->cpu_entries[cpu]);
 kfree(entries);

 return -ENOMEM;
}

int get_callchain_buffers(void)
{
 int err = 0;
 int count;

 mutex_lock(&callchain_mutex);

 count = atomic_inc_return(&nr_callchain_events);
 if (WARN_ON_ONCE(count < 1)) {
  err = -EINVAL;
  goto exit;
 }

 if (count > 1) {

  if (!callchain_cpus_entries)
   err = -ENOMEM;
  goto exit;
 }

 err = alloc_callchain_buffers();
exit:
 if (err)
  atomic_dec(&nr_callchain_events);

 mutex_unlock(&callchain_mutex);

 return err;
}

void put_callchain_buffers(void)
{
 if (atomic_dec_and_mutex_lock(&nr_callchain_events, &callchain_mutex)) {
  release_callchain_buffers();
  mutex_unlock(&callchain_mutex);
 }
}

static struct perf_callchain_entry *get_callchain_entry(int *rctx)
{
 int cpu;
 struct callchain_cpus_entries *entries;

 *rctx = get_recursion_context(this_cpu_ptr(callchain_recursion));
 if (*rctx == -1)
  return NULL;

 entries = rcu_dereference(callchain_cpus_entries);
 if (!entries)
  return NULL;

 cpu = smp_processor_id();

 return (((void *)entries->cpu_entries[cpu]) +
  (*rctx * perf_callchain_entry__sizeof()));
}

static void
put_callchain_entry(int rctx)
{
 put_recursion_context(this_cpu_ptr(callchain_recursion), rctx);
}

struct perf_callchain_entry *
perf_callchain(struct perf_event *event, struct pt_regs *regs)
{
 bool kernel = !event->attr.exclude_callchain_kernel;
 bool user = !event->attr.exclude_callchain_user;

 bool crosstask = event->ctx->task && event->ctx->task != current;

 if (!kernel && !user)
  return NULL;

 return get_perf_callchain(regs, 0, kernel, user, sysctl_perf_event_max_stack, crosstask, true);
}

struct perf_callchain_entry *
get_perf_callchain(struct pt_regs *regs, u32 init_nr, bool kernel, bool user,
     u32 max_stack, bool crosstask, bool add_mark)
{
 struct perf_callchain_entry *entry;
 struct perf_callchain_entry_ctx ctx;
 int rctx;

 entry = get_callchain_entry(&rctx);
 if (rctx == -1)
  return NULL;

 if (!entry)
  goto exit_put;

 ctx.entry = entry;
 ctx.max_stack = max_stack;
 ctx.nr = entry->nr = init_nr;
 ctx.contexts = 0;
 ctx.contexts_maxed = false;

 if (kernel && !user_mode(regs)) {
  if (add_mark)
   perf_callchain_store_context(&ctx, PERF_CONTEXT_KERNEL);
  perf_callchain_kernel(&ctx, regs);
 }

 if (user) {
  if (!user_mode(regs)) {
   if (current->mm)
    regs = task_pt_regs(current);
   else
    regs = NULL;
  }

  if (regs) {
   if (crosstask)
    goto exit_put;

   if (add_mark)
    perf_callchain_store_context(&ctx, PERF_CONTEXT_USER);
   perf_callchain_user(&ctx, regs);
  }
 }

exit_put:
 put_callchain_entry(rctx);

 return entry;
}





int perf_event_max_stack_handler(struct ctl_table *table, int write,
     void __user *buffer, size_t *lenp, loff_t *ppos)
{
 int *value = table->data;
 int new_value = *value, ret;
 struct ctl_table new_table = *table;

 new_table.data = &new_value;
 ret = proc_dointvec_minmax(&new_table, write, buffer, lenp, ppos);
 if (ret || !write)
  return ret;

 mutex_lock(&callchain_mutex);
 if (atomic_read(&nr_callchain_events))
  ret = -EBUSY;
 else
  *value = new_value;

 mutex_unlock(&callchain_mutex);

 return ret;
}






const kernel_cap_t __cap_empty_set = CAP_EMPTY_SET;
EXPORT_SYMBOL(__cap_empty_set);

int file_caps_enabled = 1;

static int __init file_caps_disable(char *str)
{
 file_caps_enabled = 0;
 return 1;
}
__setup("no_file_caps", file_caps_disable);







static void warn_legacy_capability_use(void)
{
 char name[sizeof(current->comm)];

 pr_info_once("warning: `%s' uses 32-bit capabilities (legacy support in use)\n",
       get_task_comm(name, current));
}
static void warn_deprecated_v2(void)
{
 char name[sizeof(current->comm)];

 pr_info_once("warning: `%s' uses deprecated v2 capabilities in a way that may be insecure\n",
       get_task_comm(name, current));
}





static int cap_validate_magic(cap_user_header_t header, unsigned *tocopy)
{
 __u32 version;

 if (get_user(version, &header->version))
  return -EFAULT;

 switch (version) {
 case _LINUX_CAPABILITY_VERSION_1:
  warn_legacy_capability_use();
  *tocopy = _LINUX_CAPABILITY_U32S_1;
  break;
 case _LINUX_CAPABILITY_VERSION_2:
  warn_deprecated_v2();



 case _LINUX_CAPABILITY_VERSION_3:
  *tocopy = _LINUX_CAPABILITY_U32S_3;
  break;
 default:
  if (put_user((u32)_KERNEL_CAPABILITY_VERSION, &header->version))
   return -EFAULT;
  return -EINVAL;
 }

 return 0;
}
static inline int cap_get_target_pid(pid_t pid, kernel_cap_t *pEp,
         kernel_cap_t *pIp, kernel_cap_t *pPp)
{
 int ret;

 if (pid && (pid != task_pid_vnr(current))) {
  struct task_struct *target;

  rcu_read_lock();

  target = find_task_by_vpid(pid);
  if (!target)
   ret = -ESRCH;
  else
   ret = security_capget(target, pEp, pIp, pPp);

  rcu_read_unlock();
 } else
  ret = security_capget(current, pEp, pIp, pPp);

 return ret;
}
SYSCALL_DEFINE2(capget, cap_user_header_t, header, cap_user_data_t, dataptr)
{
 int ret = 0;
 pid_t pid;
 unsigned tocopy;
 kernel_cap_t pE, pI, pP;

 ret = cap_validate_magic(header, &tocopy);
 if ((dataptr == NULL) || (ret != 0))
  return ((dataptr == NULL) && (ret == -EINVAL)) ? 0 : ret;

 if (get_user(pid, &header->pid))
  return -EFAULT;

 if (pid < 0)
  return -EINVAL;

 ret = cap_get_target_pid(pid, &pE, &pI, &pP);
 if (!ret) {
  struct __user_cap_data_struct kdata[_KERNEL_CAPABILITY_U32S];
  unsigned i;

  for (i = 0; i < tocopy; i++) {
   kdata[i].effective = pE.cap[i];
   kdata[i].permitted = pP.cap[i];
   kdata[i].inheritable = pI.cap[i];
  }
  if (copy_to_user(dataptr, kdata, tocopy
     * sizeof(struct __user_cap_data_struct))) {
   return -EFAULT;
  }
 }

 return ret;
}
SYSCALL_DEFINE2(capset, cap_user_header_t, header, const cap_user_data_t, data)
{
 struct __user_cap_data_struct kdata[_KERNEL_CAPABILITY_U32S];
 unsigned i, tocopy, copybytes;
 kernel_cap_t inheritable, permitted, effective;
 struct cred *new;
 int ret;
 pid_t pid;

 ret = cap_validate_magic(header, &tocopy);
 if (ret != 0)
  return ret;

 if (get_user(pid, &header->pid))
  return -EFAULT;


 if (pid != 0 && pid != task_pid_vnr(current))
  return -EPERM;

 copybytes = tocopy * sizeof(struct __user_cap_data_struct);
 if (copybytes > sizeof(kdata))
  return -EFAULT;

 if (copy_from_user(&kdata, data, copybytes))
  return -EFAULT;

 for (i = 0; i < tocopy; i++) {
  effective.cap[i] = kdata[i].effective;
  permitted.cap[i] = kdata[i].permitted;
  inheritable.cap[i] = kdata[i].inheritable;
 }
 while (i < _KERNEL_CAPABILITY_U32S) {
  effective.cap[i] = 0;
  permitted.cap[i] = 0;
  inheritable.cap[i] = 0;
  i++;
 }

 effective.cap[CAP_LAST_U32] &= CAP_LAST_U32_VALID_MASK;
 permitted.cap[CAP_LAST_U32] &= CAP_LAST_U32_VALID_MASK;
 inheritable.cap[CAP_LAST_U32] &= CAP_LAST_U32_VALID_MASK;

 new = prepare_creds();
 if (!new)
  return -ENOMEM;

 ret = security_capset(new, current_cred(),
         &effective, &inheritable, &permitted);
 if (ret < 0)
  goto error;

 audit_log_capset(new, current_cred());

 return commit_creds(new);

error:
 abort_creds(new);
 return ret;
}
bool has_ns_capability(struct task_struct *t,
         struct user_namespace *ns, int cap)
{
 int ret;

 rcu_read_lock();
 ret = security_capable(__task_cred(t), ns, cap);
 rcu_read_unlock();

 return (ret == 0);
}
bool has_capability(struct task_struct *t, int cap)
{
 return has_ns_capability(t, &init_user_ns, cap);
}
bool has_ns_capability_noaudit(struct task_struct *t,
          struct user_namespace *ns, int cap)
{
 int ret;

 rcu_read_lock();
 ret = security_capable_noaudit(__task_cred(t), ns, cap);
 rcu_read_unlock();

 return (ret == 0);
}
bool has_capability_noaudit(struct task_struct *t, int cap)
{
 return has_ns_capability_noaudit(t, &init_user_ns, cap);
}
bool ns_capable(struct user_namespace *ns, int cap)
{
 if (unlikely(!cap_valid(cap))) {
  pr_crit("capable() called with invalid cap=%u\n", cap);
  BUG();
 }

 if (security_capable(current_cred(), ns, cap) == 0) {
  current->flags |= PF_SUPERPRIV;
  return true;
 }
 return false;
}
EXPORT_SYMBOL(ns_capable);
bool capable(int cap)
{
 return ns_capable(&init_user_ns, cap);
}
EXPORT_SYMBOL(capable);
bool file_ns_capable(const struct file *file, struct user_namespace *ns,
       int cap)
{
 if (WARN_ON_ONCE(!cap_valid(cap)))
  return false;

 if (security_capable(file->f_cred, ns, cap) == 0)
  return true;

 return false;
}
EXPORT_SYMBOL(file_ns_capable);
bool capable_wrt_inode_uidgid(const struct inode *inode, int cap)
{
 struct user_namespace *ns = current_user_ns();

 return ns_capable(ns, cap) && kuid_has_mapping(ns, inode->i_uid) &&
  kgid_has_mapping(ns, inode->i_gid);
}
EXPORT_SYMBOL(capable_wrt_inode_uidgid);









      MAX_CFTYPE_NAME + 2)
DEFINE_MUTEX(cgroup_mutex);
DEFINE_SPINLOCK(css_set_lock);
EXPORT_SYMBOL_GPL(cgroup_mutex);
EXPORT_SYMBOL_GPL(css_set_lock);
static DEFINE_MUTEX(cgroup_mutex);
static DEFINE_SPINLOCK(css_set_lock);





static DEFINE_SPINLOCK(cgroup_idr_lock);





static DEFINE_SPINLOCK(cgroup_file_kn_lock);





static DEFINE_SPINLOCK(release_agent_path_lock);

struct percpu_rw_semaphore cgroup_threadgroup_rwsem;

 RCU_LOCKDEP_WARN(!rcu_read_lock_held() && \
      !lockdep_is_held(&cgroup_mutex), \
      "cgroup_mutex or RCU read lock required");







static struct workqueue_struct *cgroup_destroy_wq;





static struct workqueue_struct *cgroup_pidlist_destroy_wq;


static struct cgroup_subsys *cgroup_subsys[] = {
};


static const char *cgroup_subsys_name[] = {
};


 DEFINE_STATIC_KEY_TRUE(_x ## _cgrp_subsys_enabled_key); \
 DEFINE_STATIC_KEY_TRUE(_x ## _cgrp_subsys_on_dfl_key); \
 EXPORT_SYMBOL_GPL(_x ## _cgrp_subsys_enabled_key); \
 EXPORT_SYMBOL_GPL(_x ## _cgrp_subsys_on_dfl_key);

static struct static_key_true *cgroup_subsys_enabled_key[] = {
};

static struct static_key_true *cgroup_subsys_on_dfl_key[] = {
};






struct cgroup_root cgrp_dfl_root;
EXPORT_SYMBOL_GPL(cgrp_dfl_root);





static bool cgrp_dfl_visible;


static u16 cgroup_no_v1_mask;


static u16 cgrp_dfl_inhibit_ss_mask;


static unsigned long cgrp_dfl_implicit_ss_mask;



static LIST_HEAD(cgroup_roots);
static int cgroup_root_count;


static DEFINE_IDR(cgroup_hierarchy_idr);
static u64 css_serial_nr_next = 1;






static u16 have_fork_callback __read_mostly;
static u16 have_exit_callback __read_mostly;
static u16 have_free_callback __read_mostly;


struct cgroup_namespace init_cgroup_ns = {
 .count = { .counter = 2, },
 .user_ns = &init_user_ns,
 .ns.ops = &cgroupns_operations,
 .ns.inum = PROC_CGROUP_INIT_INO,
 .root_cset = &init_css_set,
};


static u16 have_canfork_callback __read_mostly;

static struct file_system_type cgroup2_fs_type;
static struct cftype cgroup_dfl_base_files[];
static struct cftype cgroup_legacy_base_files[];

static int rebind_subsystems(struct cgroup_root *dst_root, u16 ss_mask);
static void cgroup_lock_and_drain_offline(struct cgroup *cgrp);
static int cgroup_apply_control(struct cgroup *cgrp);
static void cgroup_finalize_control(struct cgroup *cgrp, int ret);
static void css_task_iter_advance(struct css_task_iter *it);
static int cgroup_destroy_locked(struct cgroup *cgrp);
static struct cgroup_subsys_state *css_create(struct cgroup *cgrp,
           struct cgroup_subsys *ss);
static void css_release(struct percpu_ref *ref);
static void kill_css(struct cgroup_subsys_state *css);
static int cgroup_addrm_files(struct cgroup_subsys_state *css,
         struct cgroup *cgrp, struct cftype cfts[],
         bool is_add);
static bool cgroup_ssid_enabled(int ssid)
{
 if (CGROUP_SUBSYS_COUNT == 0)
  return false;

 return static_key_enabled(cgroup_subsys_enabled_key[ssid]);
}

static bool cgroup_ssid_no_v1(int ssid)
{
 return cgroup_no_v1_mask & (1 << ssid);
}
static bool cgroup_on_dfl(const struct cgroup *cgrp)
{
 return cgrp->root == &cgrp_dfl_root;
}


static int cgroup_idr_alloc(struct idr *idr, void *ptr, int start, int end,
       gfp_t gfp_mask)
{
 int ret;

 idr_preload(gfp_mask);
 spin_lock_bh(&cgroup_idr_lock);
 ret = idr_alloc(idr, ptr, start, end, gfp_mask & ~__GFP_DIRECT_RECLAIM);
 spin_unlock_bh(&cgroup_idr_lock);
 idr_preload_end();
 return ret;
}

static void *cgroup_idr_replace(struct idr *idr, void *ptr, int id)
{
 void *ret;

 spin_lock_bh(&cgroup_idr_lock);
 ret = idr_replace(idr, ptr, id);
 spin_unlock_bh(&cgroup_idr_lock);
 return ret;
}

static void cgroup_idr_remove(struct idr *idr, int id)
{
 spin_lock_bh(&cgroup_idr_lock);
 idr_remove(idr, id);
 spin_unlock_bh(&cgroup_idr_lock);
}

static struct cgroup *cgroup_parent(struct cgroup *cgrp)
{
 struct cgroup_subsys_state *parent_css = cgrp->self.parent;

 if (parent_css)
  return container_of(parent_css, struct cgroup, self);
 return NULL;
}


static u16 cgroup_control(struct cgroup *cgrp)
{
 struct cgroup *parent = cgroup_parent(cgrp);
 u16 root_ss_mask = cgrp->root->subsys_mask;

 if (parent)
  return parent->subtree_control;

 if (cgroup_on_dfl(cgrp))
  root_ss_mask &= ~(cgrp_dfl_inhibit_ss_mask |
      cgrp_dfl_implicit_ss_mask);
 return root_ss_mask;
}


static u16 cgroup_ss_mask(struct cgroup *cgrp)
{
 struct cgroup *parent = cgroup_parent(cgrp);

 if (parent)
  return parent->subtree_ss_mask;

 return cgrp->root->subsys_mask;
}
static struct cgroup_subsys_state *cgroup_css(struct cgroup *cgrp,
           struct cgroup_subsys *ss)
{
 if (ss)
  return rcu_dereference_check(cgrp->subsys[ss->id],
     lockdep_is_held(&cgroup_mutex));
 else
  return &cgrp->self;
}
static struct cgroup_subsys_state *cgroup_e_css(struct cgroup *cgrp,
      struct cgroup_subsys *ss)
{
 lockdep_assert_held(&cgroup_mutex);

 if (!ss)
  return &cgrp->self;





 while (!(cgroup_ss_mask(cgrp) & (1 << ss->id))) {
  cgrp = cgroup_parent(cgrp);
  if (!cgrp)
   return NULL;
 }

 return cgroup_css(cgrp, ss);
}
struct cgroup_subsys_state *cgroup_get_e_css(struct cgroup *cgrp,
          struct cgroup_subsys *ss)
{
 struct cgroup_subsys_state *css;

 rcu_read_lock();

 do {
  css = cgroup_css(cgrp, ss);

  if (css && css_tryget_online(css))
   goto out_unlock;
  cgrp = cgroup_parent(cgrp);
 } while (cgrp);

 css = init_css_set.subsys[ss->id];
 css_get(css);
out_unlock:
 rcu_read_unlock();
 return css;
}


static inline bool cgroup_is_dead(const struct cgroup *cgrp)
{
 return !(cgrp->self.flags & CSS_ONLINE);
}

static void cgroup_get(struct cgroup *cgrp)
{
 WARN_ON_ONCE(cgroup_is_dead(cgrp));
 css_get(&cgrp->self);
}

static bool cgroup_tryget(struct cgroup *cgrp)
{
 return css_tryget(&cgrp->self);
}

struct cgroup_subsys_state *of_css(struct kernfs_open_file *of)
{
 struct cgroup *cgrp = of->kn->parent->priv;
 struct cftype *cft = of_cft(of);
 if (cft->ss)
  return rcu_dereference_raw(cgrp->subsys[cft->ss->id]);
 else
  return &cgrp->self;
}
EXPORT_SYMBOL_GPL(of_css);

static int notify_on_release(const struct cgroup *cgrp)
{
 return test_bit(CGRP_NOTIFY_ON_RELEASE, &cgrp->flags);
}
 for ((ssid) = 0; (ssid) < CGROUP_SUBSYS_COUNT; (ssid)++) \
  if (!((css) = rcu_dereference_check( \
    (cgrp)->subsys[(ssid)], \
    lockdep_is_held(&cgroup_mutex)))) { } \
  else
 for ((ssid) = 0; (ssid) < CGROUP_SUBSYS_COUNT; (ssid)++) \
  if (!((css) = cgroup_e_css(cgrp, cgroup_subsys[(ssid)]))) \
   ; \
  else






 for ((ssid) = 0; (ssid) < CGROUP_SUBSYS_COUNT && \
      (((ss) = cgroup_subsys[ssid]) || true); (ssid)++)
 unsigned long __ss_mask = (ss_mask); \
 if (!CGROUP_SUBSYS_COUNT) { \
  (ssid) = 0; \
  break; \
 } \
 for_each_set_bit(ssid, &__ss_mask, CGROUP_SUBSYS_COUNT) { \
  (ss) = cgroup_subsys[ssid]; \
  {

  } \
 } \
} while (false)


 list_for_each_entry((root), &cgroup_roots, root_list)


 list_for_each_entry((child), &(cgrp)->self.children, self.sibling) \
  if (({ lockdep_assert_held(&cgroup_mutex); \
         cgroup_is_dead(child); })) \
   ; \
  else


 css_for_each_descendant_pre((d_css), cgroup_css((cgrp), NULL)) \
  if (({ lockdep_assert_held(&cgroup_mutex); \
         (dsct) = (d_css)->cgroup; \
         cgroup_is_dead(dsct); })) \
   ; \
  else


 css_for_each_descendant_post((d_css), cgroup_css((cgrp), NULL)) \
  if (({ lockdep_assert_held(&cgroup_mutex); \
         (dsct) = (d_css)->cgroup; \
         cgroup_is_dead(dsct); })) \
   ; \
  else

static void cgroup_release_agent(struct work_struct *work);
static void check_for_release(struct cgroup *cgrp);
struct cgrp_cset_link {

 struct cgroup *cgrp;
 struct css_set *cset;


 struct list_head cset_link;


 struct list_head cgrp_link;
};
struct css_set init_css_set = {
 .refcount = ATOMIC_INIT(1),
 .cgrp_links = LIST_HEAD_INIT(init_css_set.cgrp_links),
 .tasks = LIST_HEAD_INIT(init_css_set.tasks),
 .mg_tasks = LIST_HEAD_INIT(init_css_set.mg_tasks),
 .mg_preload_node = LIST_HEAD_INIT(init_css_set.mg_preload_node),
 .mg_node = LIST_HEAD_INIT(init_css_set.mg_node),
 .task_iters = LIST_HEAD_INIT(init_css_set.task_iters),
};

static int css_set_count = 1;





static bool css_set_populated(struct css_set *cset)
{
 lockdep_assert_held(&css_set_lock);

 return !list_empty(&cset->tasks) || !list_empty(&cset->mg_tasks);
}
static void cgroup_update_populated(struct cgroup *cgrp, bool populated)
{
 lockdep_assert_held(&css_set_lock);

 do {
  bool trigger;

  if (populated)
   trigger = !cgrp->populated_cnt++;
  else
   trigger = !--cgrp->populated_cnt;

  if (!trigger)
   break;

  check_for_release(cgrp);
  cgroup_file_notify(&cgrp->events_file);

  cgrp = cgroup_parent(cgrp);
 } while (cgrp);
}
static void css_set_update_populated(struct css_set *cset, bool populated)
{
 struct cgrp_cset_link *link;

 lockdep_assert_held(&css_set_lock);

 list_for_each_entry(link, &cset->cgrp_links, cgrp_link)
  cgroup_update_populated(link->cgrp, populated);
}
static void css_set_move_task(struct task_struct *task,
         struct css_set *from_cset, struct css_set *to_cset,
         bool use_mg_tasks)
{
 lockdep_assert_held(&css_set_lock);

 if (to_cset && !css_set_populated(to_cset))
  css_set_update_populated(to_cset, true);

 if (from_cset) {
  struct css_task_iter *it, *pos;

  WARN_ON_ONCE(list_empty(&task->cg_list));
  list_for_each_entry_safe(it, pos, &from_cset->task_iters,
      iters_node)
   if (it->task_pos == &task->cg_list)
    css_task_iter_advance(it);

  list_del_init(&task->cg_list);
  if (!css_set_populated(from_cset))
   css_set_update_populated(from_cset, false);
 } else {
  WARN_ON_ONCE(!list_empty(&task->cg_list));
 }

 if (to_cset) {






  WARN_ON_ONCE(task->flags & PF_EXITING);

  rcu_assign_pointer(task->cgroups, to_cset);
  list_add_tail(&task->cg_list, use_mg_tasks ? &to_cset->mg_tasks :
            &to_cset->tasks);
 }
}






static DEFINE_HASHTABLE(css_set_table, CSS_SET_HASH_BITS);

static unsigned long css_set_hash(struct cgroup_subsys_state *css[])
{
 unsigned long key = 0UL;
 struct cgroup_subsys *ss;
 int i;

 for_each_subsys(ss, i)
  key += (unsigned long)css[i];
 key = (key >> 16) ^ key;

 return key;
}

static void put_css_set_locked(struct css_set *cset)
{
 struct cgrp_cset_link *link, *tmp_link;
 struct cgroup_subsys *ss;
 int ssid;

 lockdep_assert_held(&css_set_lock);

 if (!atomic_dec_and_test(&cset->refcount))
  return;


 for_each_subsys(ss, ssid) {
  list_del(&cset->e_cset_node[ssid]);
  css_put(cset->subsys[ssid]);
 }
 hash_del(&cset->hlist);
 css_set_count--;

 list_for_each_entry_safe(link, tmp_link, &cset->cgrp_links, cgrp_link) {
  list_del(&link->cset_link);
  list_del(&link->cgrp_link);
  if (cgroup_parent(link->cgrp))
   cgroup_put(link->cgrp);
  kfree(link);
 }

 kfree_rcu(cset, rcu_head);
}

static void put_css_set(struct css_set *cset)
{
 unsigned long flags;






 if (atomic_add_unless(&cset->refcount, -1, 1))
  return;

 spin_lock_irqsave(&css_set_lock, flags);
 put_css_set_locked(cset);
 spin_unlock_irqrestore(&css_set_lock, flags);
}




static inline void get_css_set(struct css_set *cset)
{
 atomic_inc(&cset->refcount);
}
static bool compare_css_sets(struct css_set *cset,
        struct css_set *old_cset,
        struct cgroup *new_cgrp,
        struct cgroup_subsys_state *template[])
{
 struct list_head *l1, *l2;






 if (memcmp(template, cset->subsys, sizeof(cset->subsys)))
  return false;







 l1 = &cset->cgrp_links;
 l2 = &old_cset->cgrp_links;
 while (1) {
  struct cgrp_cset_link *link1, *link2;
  struct cgroup *cgrp1, *cgrp2;

  l1 = l1->next;
  l2 = l2->next;

  if (l1 == &cset->cgrp_links) {
   BUG_ON(l2 != &old_cset->cgrp_links);
   break;
  } else {
   BUG_ON(l2 == &old_cset->cgrp_links);
  }

  link1 = list_entry(l1, struct cgrp_cset_link, cgrp_link);
  link2 = list_entry(l2, struct cgrp_cset_link, cgrp_link);
  cgrp1 = link1->cgrp;
  cgrp2 = link2->cgrp;

  BUG_ON(cgrp1->root != cgrp2->root);
  if (cgrp1->root == new_cgrp->root) {
   if (cgrp1 != new_cgrp)
    return false;
  } else {
   if (cgrp1 != cgrp2)
    return false;
  }
 }
 return true;
}







static struct css_set *find_existing_css_set(struct css_set *old_cset,
     struct cgroup *cgrp,
     struct cgroup_subsys_state *template[])
{
 struct cgroup_root *root = cgrp->root;
 struct cgroup_subsys *ss;
 struct css_set *cset;
 unsigned long key;
 int i;






 for_each_subsys(ss, i) {
  if (root->subsys_mask & (1UL << i)) {




   template[i] = cgroup_e_css(cgrp, ss);
  } else {




   template[i] = old_cset->subsys[i];
  }
 }

 key = css_set_hash(template);
 hash_for_each_possible(css_set_table, cset, hlist, key) {
  if (!compare_css_sets(cset, old_cset, cgrp, template))
   continue;


  return cset;
 }


 return NULL;
}

static void free_cgrp_cset_links(struct list_head *links_to_free)
{
 struct cgrp_cset_link *link, *tmp_link;

 list_for_each_entry_safe(link, tmp_link, links_to_free, cset_link) {
  list_del(&link->cset_link);
  kfree(link);
 }
}
static int allocate_cgrp_cset_links(int count, struct list_head *tmp_links)
{
 struct cgrp_cset_link *link;
 int i;

 INIT_LIST_HEAD(tmp_links);

 for (i = 0; i < count; i++) {
  link = kzalloc(sizeof(*link), GFP_KERNEL);
  if (!link) {
   free_cgrp_cset_links(tmp_links);
   return -ENOMEM;
  }
  list_add(&link->cset_link, tmp_links);
 }
 return 0;
}







static void link_css_set(struct list_head *tmp_links, struct css_set *cset,
    struct cgroup *cgrp)
{
 struct cgrp_cset_link *link;

 BUG_ON(list_empty(tmp_links));

 if (cgroup_on_dfl(cgrp))
  cset->dfl_cgrp = cgrp;

 link = list_first_entry(tmp_links, struct cgrp_cset_link, cset_link);
 link->cset = cset;
 link->cgrp = cgrp;





 list_move_tail(&link->cset_link, &cgrp->cset_links);
 list_add_tail(&link->cgrp_link, &cset->cgrp_links);

 if (cgroup_parent(cgrp))
  cgroup_get(cgrp);
}
static struct css_set *find_css_set(struct css_set *old_cset,
        struct cgroup *cgrp)
{
 struct cgroup_subsys_state *template[CGROUP_SUBSYS_COUNT] = { };
 struct css_set *cset;
 struct list_head tmp_links;
 struct cgrp_cset_link *link;
 struct cgroup_subsys *ss;
 unsigned long key;
 int ssid;

 lockdep_assert_held(&cgroup_mutex);



 spin_lock_irq(&css_set_lock);
 cset = find_existing_css_set(old_cset, cgrp, template);
 if (cset)
  get_css_set(cset);
 spin_unlock_irq(&css_set_lock);

 if (cset)
  return cset;

 cset = kzalloc(sizeof(*cset), GFP_KERNEL);
 if (!cset)
  return NULL;


 if (allocate_cgrp_cset_links(cgroup_root_count, &tmp_links) < 0) {
  kfree(cset);
  return NULL;
 }

 atomic_set(&cset->refcount, 1);
 INIT_LIST_HEAD(&cset->cgrp_links);
 INIT_LIST_HEAD(&cset->tasks);
 INIT_LIST_HEAD(&cset->mg_tasks);
 INIT_LIST_HEAD(&cset->mg_preload_node);
 INIT_LIST_HEAD(&cset->mg_node);
 INIT_LIST_HEAD(&cset->task_iters);
 INIT_HLIST_NODE(&cset->hlist);



 memcpy(cset->subsys, template, sizeof(cset->subsys));

 spin_lock_irq(&css_set_lock);

 list_for_each_entry(link, &old_cset->cgrp_links, cgrp_link) {
  struct cgroup *c = link->cgrp;

  if (c->root == cgrp->root)
   c = cgrp;
  link_css_set(&tmp_links, cset, c);
 }

 BUG_ON(!list_empty(&tmp_links));

 css_set_count++;


 key = css_set_hash(cset->subsys);
 hash_add(css_set_table, &cset->hlist, key);

 for_each_subsys(ss, ssid) {
  struct cgroup_subsys_state *css = cset->subsys[ssid];

  list_add_tail(&cset->e_cset_node[ssid],
         &css->cgroup->e_csets[ssid]);
  css_get(css);
 }

 spin_unlock_irq(&css_set_lock);

 return cset;
}

static struct cgroup_root *cgroup_root_from_kf(struct kernfs_root *kf_root)
{
 struct cgroup *root_cgrp = kf_root->kn->priv;

 return root_cgrp->root;
}

static int cgroup_init_root_id(struct cgroup_root *root)
{
 int id;

 lockdep_assert_held(&cgroup_mutex);

 id = idr_alloc_cyclic(&cgroup_hierarchy_idr, root, 0, 0, GFP_KERNEL);
 if (id < 0)
  return id;

 root->hierarchy_id = id;
 return 0;
}

static void cgroup_exit_root_id(struct cgroup_root *root)
{
 lockdep_assert_held(&cgroup_mutex);

 if (root->hierarchy_id) {
  idr_remove(&cgroup_hierarchy_idr, root->hierarchy_id);
  root->hierarchy_id = 0;
 }
}

static void cgroup_free_root(struct cgroup_root *root)
{
 if (root) {

  WARN_ON_ONCE(root->hierarchy_id);

  idr_destroy(&root->cgroup_idr);
  kfree(root);
 }
}

static void cgroup_destroy_root(struct cgroup_root *root)
{
 struct cgroup *cgrp = &root->cgrp;
 struct cgrp_cset_link *link, *tmp_link;

 cgroup_lock_and_drain_offline(&cgrp_dfl_root.cgrp);

 BUG_ON(atomic_read(&root->nr_cgrps));
 BUG_ON(!list_empty(&cgrp->self.children));


 WARN_ON(rebind_subsystems(&cgrp_dfl_root, root->subsys_mask));





 spin_lock_irq(&css_set_lock);

 list_for_each_entry_safe(link, tmp_link, &cgrp->cset_links, cset_link) {
  list_del(&link->cset_link);
  list_del(&link->cgrp_link);
  kfree(link);
 }

 spin_unlock_irq(&css_set_lock);

 if (!list_empty(&root->root_list)) {
  list_del(&root->root_list);
  cgroup_root_count--;
 }

 cgroup_exit_root_id(root);

 mutex_unlock(&cgroup_mutex);

 kernfs_destroy_root(root->kf_root);
 cgroup_free_root(root);
}





static struct cgroup *
current_cgns_cgroup_from_root(struct cgroup_root *root)
{
 struct cgroup *res = NULL;
 struct css_set *cset;

 lockdep_assert_held(&css_set_lock);

 rcu_read_lock();

 cset = current->nsproxy->cgroup_ns->root_cset;
 if (cset == &init_css_set) {
  res = &root->cgrp;
 } else {
  struct cgrp_cset_link *link;

  list_for_each_entry(link, &cset->cgrp_links, cgrp_link) {
   struct cgroup *c = link->cgrp;

   if (c->root == root) {
    res = c;
    break;
   }
  }
 }
 rcu_read_unlock();

 BUG_ON(!res);
 return res;
}


static struct cgroup *cset_cgroup_from_root(struct css_set *cset,
         struct cgroup_root *root)
{
 struct cgroup *res = NULL;

 lockdep_assert_held(&cgroup_mutex);
 lockdep_assert_held(&css_set_lock);

 if (cset == &init_css_set) {
  res = &root->cgrp;
 } else {
  struct cgrp_cset_link *link;

  list_for_each_entry(link, &cset->cgrp_links, cgrp_link) {
   struct cgroup *c = link->cgrp;

   if (c->root == root) {
    res = c;
    break;
   }
  }
 }

 BUG_ON(!res);
 return res;
}





static struct cgroup *task_cgroup_from_root(struct task_struct *task,
         struct cgroup_root *root)
{





 return cset_cgroup_from_root(task_css_set(task), root);
}
static struct kernfs_syscall_ops cgroup_kf_syscall_ops;
static const struct file_operations proc_cgroupstats_operations;

static char *cgroup_file_name(struct cgroup *cgrp, const struct cftype *cft,
         char *buf)
{
 struct cgroup_subsys *ss = cft->ss;

 if (cft->ss && !(cft->flags & CFTYPE_NO_PREFIX) &&
     !(cgrp->root->flags & CGRP_ROOT_NOPREFIX))
  snprintf(buf, CGROUP_FILE_NAME_MAX, "%s.%s",
    cgroup_on_dfl(cgrp) ? ss->name : ss->legacy_name,
    cft->name);
 else
  strncpy(buf, cft->name, CGROUP_FILE_NAME_MAX);
 return buf;
}







static umode_t cgroup_file_mode(const struct cftype *cft)
{
 umode_t mode = 0;

 if (cft->read_u64 || cft->read_s64 || cft->seq_show)
  mode |= S_IRUGO;

 if (cft->write_u64 || cft->write_s64 || cft->write) {
  if (cft->flags & CFTYPE_WORLD_WRITABLE)
   mode |= S_IWUGO;
  else
   mode |= S_IWUSR;
 }

 return mode;
}
static u16 cgroup_calc_subtree_ss_mask(u16 subtree_control, u16 this_ss_mask)
{
 u16 cur_ss_mask = subtree_control;
 struct cgroup_subsys *ss;
 int ssid;

 lockdep_assert_held(&cgroup_mutex);

 cur_ss_mask |= cgrp_dfl_implicit_ss_mask;

 while (true) {
  u16 new_ss_mask = cur_ss_mask;

  do_each_subsys_mask(ss, ssid, cur_ss_mask) {
   new_ss_mask |= ss->depends_on;
  } while_each_subsys_mask();






  new_ss_mask &= this_ss_mask;

  if (new_ss_mask == cur_ss_mask)
   break;
  cur_ss_mask = new_ss_mask;
 }

 return cur_ss_mask;
}
static void cgroup_kn_unlock(struct kernfs_node *kn)
{
 struct cgroup *cgrp;

 if (kernfs_type(kn) == KERNFS_DIR)
  cgrp = kn->priv;
 else
  cgrp = kn->parent->priv;

 mutex_unlock(&cgroup_mutex);

 kernfs_unbreak_active_protection(kn);
 cgroup_put(cgrp);
}
static struct cgroup *cgroup_kn_lock_live(struct kernfs_node *kn,
       bool drain_offline)
{
 struct cgroup *cgrp;

 if (kernfs_type(kn) == KERNFS_DIR)
  cgrp = kn->priv;
 else
  cgrp = kn->parent->priv;







 if (!cgroup_tryget(cgrp))
  return NULL;
 kernfs_break_active_protection(kn);

 if (drain_offline)
  cgroup_lock_and_drain_offline(cgrp);
 else
  mutex_lock(&cgroup_mutex);

 if (!cgroup_is_dead(cgrp))
  return cgrp;

 cgroup_kn_unlock(kn);
 return NULL;
}

static void cgroup_rm_file(struct cgroup *cgrp, const struct cftype *cft)
{
 char name[CGROUP_FILE_NAME_MAX];

 lockdep_assert_held(&cgroup_mutex);

 if (cft->file_offset) {
  struct cgroup_subsys_state *css = cgroup_css(cgrp, cft->ss);
  struct cgroup_file *cfile = (void *)css + cft->file_offset;

  spin_lock_irq(&cgroup_file_kn_lock);
  cfile->kn = NULL;
  spin_unlock_irq(&cgroup_file_kn_lock);
 }

 kernfs_remove_by_name(cgrp->kn, cgroup_file_name(cgrp, cft, name));
}





static void css_clear_dir(struct cgroup_subsys_state *css)
{
 struct cgroup *cgrp = css->cgroup;
 struct cftype *cfts;

 if (!(css->flags & CSS_VISIBLE))
  return;

 css->flags &= ~CSS_VISIBLE;

 list_for_each_entry(cfts, &css->ss->cfts, node)
  cgroup_addrm_files(css, cgrp, cfts, false);
}







static int css_populate_dir(struct cgroup_subsys_state *css)
{
 struct cgroup *cgrp = css->cgroup;
 struct cftype *cfts, *failed_cfts;
 int ret;

 if ((css->flags & CSS_VISIBLE) || !cgrp->kn)
  return 0;

 if (!css->ss) {
  if (cgroup_on_dfl(cgrp))
   cfts = cgroup_dfl_base_files;
  else
   cfts = cgroup_legacy_base_files;

  return cgroup_addrm_files(&cgrp->self, cgrp, cfts, true);
 }

 list_for_each_entry(cfts, &css->ss->cfts, node) {
  ret = cgroup_addrm_files(css, cgrp, cfts, true);
  if (ret < 0) {
   failed_cfts = cfts;
   goto err;
  }
 }

 css->flags |= CSS_VISIBLE;

 return 0;
err:
 list_for_each_entry(cfts, &css->ss->cfts, node) {
  if (cfts == failed_cfts)
   break;
  cgroup_addrm_files(css, cgrp, cfts, false);
 }
 return ret;
}

static int rebind_subsystems(struct cgroup_root *dst_root, u16 ss_mask)
{
 struct cgroup *dcgrp = &dst_root->cgrp;
 struct cgroup_subsys *ss;
 int ssid, i, ret;

 lockdep_assert_held(&cgroup_mutex);

 do_each_subsys_mask(ss, ssid, ss_mask) {





  if (css_next_child(NULL, cgroup_css(&ss->root->cgrp, ss)) &&
      !ss->implicit_on_dfl)
   return -EBUSY;


  if (ss->root != &cgrp_dfl_root && dst_root != &cgrp_dfl_root)
   return -EBUSY;
 } while_each_subsys_mask();

 do_each_subsys_mask(ss, ssid, ss_mask) {
  struct cgroup_root *src_root = ss->root;
  struct cgroup *scgrp = &src_root->cgrp;
  struct cgroup_subsys_state *css = cgroup_css(scgrp, ss);
  struct css_set *cset;

  WARN_ON(!css || cgroup_css(dcgrp, ss));


  src_root->subsys_mask &= ~(1 << ssid);
  WARN_ON(cgroup_apply_control(scgrp));
  cgroup_finalize_control(scgrp, 0);


  RCU_INIT_POINTER(scgrp->subsys[ssid], NULL);
  rcu_assign_pointer(dcgrp->subsys[ssid], css);
  ss->root = dst_root;
  css->cgroup = dcgrp;

  spin_lock_irq(&css_set_lock);
  hash_for_each(css_set_table, i, cset, hlist)
   list_move_tail(&cset->e_cset_node[ss->id],
           &dcgrp->e_csets[ss->id]);
  spin_unlock_irq(&css_set_lock);


  dst_root->subsys_mask |= 1 << ssid;
  if (dst_root == &cgrp_dfl_root) {
   static_branch_enable(cgroup_subsys_on_dfl_key[ssid]);
  } else {
   dcgrp->subtree_control |= 1 << ssid;
   static_branch_disable(cgroup_subsys_on_dfl_key[ssid]);
  }

  ret = cgroup_apply_control(dcgrp);
  if (ret)
   pr_warn("partial failure to rebind %s controller (err=%d)\n",
    ss->name, ret);

  if (ss->bind)
   ss->bind(css);
 } while_each_subsys_mask();

 kernfs_activate(dcgrp->kn);
 return 0;
}

static int cgroup_show_path(struct seq_file *sf, struct kernfs_node *kf_node,
       struct kernfs_root *kf_root)
{
 int len = 0;
 char *buf = NULL;
 struct cgroup_root *kf_cgroot = cgroup_root_from_kf(kf_root);
 struct cgroup *ns_cgroup;

 buf = kmalloc(PATH_MAX, GFP_KERNEL);
 if (!buf)
  return -ENOMEM;

 spin_lock_irq(&css_set_lock);
 ns_cgroup = current_cgns_cgroup_from_root(kf_cgroot);
 len = kernfs_path_from_node(kf_node, ns_cgroup->kn, buf, PATH_MAX);
 spin_unlock_irq(&css_set_lock);

 if (len >= PATH_MAX)
  len = -ERANGE;
 else if (len > 0) {
  seq_escape(sf, buf, " \t\n\\");
  len = 0;
 }
 kfree(buf);
 return len;
}

static int cgroup_show_options(struct seq_file *seq,
          struct kernfs_root *kf_root)
{
 struct cgroup_root *root = cgroup_root_from_kf(kf_root);
 struct cgroup_subsys *ss;
 int ssid;

 if (root != &cgrp_dfl_root)
  for_each_subsys(ss, ssid)
   if (root->subsys_mask & (1 << ssid))
    seq_show_option(seq, ss->legacy_name, NULL);
 if (root->flags & CGRP_ROOT_NOPREFIX)
  seq_puts(seq, ",noprefix");
 if (root->flags & CGRP_ROOT_XATTR)
  seq_puts(seq, ",xattr");

 spin_lock(&release_agent_path_lock);
 if (strlen(root->release_agent_path))
  seq_show_option(seq, "release_agent",
    root->release_agent_path);
 spin_unlock(&release_agent_path_lock);

 if (test_bit(CGRP_CPUSET_CLONE_CHILDREN, &root->cgrp.flags))
  seq_puts(seq, ",clone_children");
 if (strlen(root->name))
  seq_show_option(seq, "name", root->name);
 return 0;
}

struct cgroup_sb_opts {
 u16 subsys_mask;
 unsigned int flags;
 char *release_agent;
 bool cpuset_clone_children;
 char *name;

 bool none;
};

static int parse_cgroupfs_options(char *data, struct cgroup_sb_opts *opts)
{
 char *token, *o = data;
 bool all_ss = false, one_ss = false;
 u16 mask = U16_MAX;
 struct cgroup_subsys *ss;
 int nr_opts = 0;
 int i;

 mask = ~((u16)1 << cpuset_cgrp_id);

 memset(opts, 0, sizeof(*opts));

 while ((token = strsep(&o, ",")) != NULL) {
  nr_opts++;

  if (!*token)
   return -EINVAL;
  if (!strcmp(token, "none")) {

   opts->none = true;
   continue;
  }
  if (!strcmp(token, "all")) {

   if (one_ss)
    return -EINVAL;
   all_ss = true;
   continue;
  }
  if (!strcmp(token, "noprefix")) {
   opts->flags |= CGRP_ROOT_NOPREFIX;
   continue;
  }
  if (!strcmp(token, "clone_children")) {
   opts->cpuset_clone_children = true;
   continue;
  }
  if (!strcmp(token, "xattr")) {
   opts->flags |= CGRP_ROOT_XATTR;
   continue;
  }
  if (!strncmp(token, "release_agent=", 14)) {

   if (opts->release_agent)
    return -EINVAL;
   opts->release_agent =
    kstrndup(token + 14, PATH_MAX - 1, GFP_KERNEL);
   if (!opts->release_agent)
    return -ENOMEM;
   continue;
  }
  if (!strncmp(token, "name=", 5)) {
   const char *name = token + 5;

   if (!strlen(name))
    return -EINVAL;

   for (i = 0; i < strlen(name); i++) {
    char c = name[i];
    if (isalnum(c))
     continue;
    if ((c == '.') || (c == '-') || (c == '_'))
     continue;
    return -EINVAL;
   }

   if (opts->name)
    return -EINVAL;
   opts->name = kstrndup(name,
           MAX_CGROUP_ROOT_NAMELEN - 1,
           GFP_KERNEL);
   if (!opts->name)
    return -ENOMEM;

   continue;
  }

  for_each_subsys(ss, i) {
   if (strcmp(token, ss->legacy_name))
    continue;
   if (!cgroup_ssid_enabled(i))
    continue;
   if (cgroup_ssid_no_v1(i))
    continue;


   if (all_ss)
    return -EINVAL;
   opts->subsys_mask |= (1 << i);
   one_ss = true;

   break;
  }
  if (i == CGROUP_SUBSYS_COUNT)
   return -ENOENT;
 }






 if (all_ss || (!one_ss && !opts->none && !opts->name))
  for_each_subsys(ss, i)
   if (cgroup_ssid_enabled(i) && !cgroup_ssid_no_v1(i))
    opts->subsys_mask |= (1 << i);





 if (!opts->subsys_mask && !opts->name)
  return -EINVAL;






 if ((opts->flags & CGRP_ROOT_NOPREFIX) && (opts->subsys_mask & mask))
  return -EINVAL;


 if (opts->subsys_mask && opts->none)
  return -EINVAL;

 return 0;
}

static int cgroup_remount(struct kernfs_root *kf_root, int *flags, char *data)
{
 int ret = 0;
 struct cgroup_root *root = cgroup_root_from_kf(kf_root);
 struct cgroup_sb_opts opts;
 u16 added_mask, removed_mask;

 if (root == &cgrp_dfl_root) {
  pr_err("remount is not allowed\n");
  return -EINVAL;
 }

 cgroup_lock_and_drain_offline(&cgrp_dfl_root.cgrp);


 ret = parse_cgroupfs_options(data, &opts);
 if (ret)
  goto out_unlock;

 if (opts.subsys_mask != root->subsys_mask || opts.release_agent)
  pr_warn("option changes via remount are deprecated (pid=%d comm=%s)\n",
   task_tgid_nr(current), current->comm);

 added_mask = opts.subsys_mask & ~root->subsys_mask;
 removed_mask = root->subsys_mask & ~opts.subsys_mask;


 if ((opts.flags ^ root->flags) ||
     (opts.name && strcmp(opts.name, root->name))) {
  pr_err("option or name mismatch, new: 0x%x \"%s\", old: 0x%x \"%s\"\n",
         opts.flags, opts.name ?: "", root->flags, root->name);
  ret = -EINVAL;
  goto out_unlock;
 }


 if (!list_empty(&root->cgrp.self.children)) {
  ret = -EBUSY;
  goto out_unlock;
 }

 ret = rebind_subsystems(root, added_mask);
 if (ret)
  goto out_unlock;

 WARN_ON(rebind_subsystems(&cgrp_dfl_root, removed_mask));

 if (opts.release_agent) {
  spin_lock(&release_agent_path_lock);
  strcpy(root->release_agent_path, opts.release_agent);
  spin_unlock(&release_agent_path_lock);
 }
 out_unlock:
 kfree(opts.release_agent);
 kfree(opts.name);
 mutex_unlock(&cgroup_mutex);
 return ret;
}







static bool use_task_css_set_links __read_mostly;

static void cgroup_enable_task_cg_lists(void)
{
 struct task_struct *p, *g;

 spin_lock_irq(&css_set_lock);

 if (use_task_css_set_links)
  goto out_unlock;

 use_task_css_set_links = true;
 read_lock(&tasklist_lock);
 do_each_thread(g, p) {
  WARN_ON_ONCE(!list_empty(&p->cg_list) ||
        task_css_set(p) != &init_css_set);
  spin_lock(&p->sighand->siglock);
  if (!(p->flags & PF_EXITING)) {
   struct css_set *cset = task_css_set(p);

   if (!css_set_populated(cset))
    css_set_update_populated(cset, true);
   list_add_tail(&p->cg_list, &cset->tasks);
   get_css_set(cset);
  }
  spin_unlock(&p->sighand->siglock);
 } while_each_thread(g, p);
 read_unlock(&tasklist_lock);
out_unlock:
 spin_unlock_irq(&css_set_lock);
}

static void init_cgroup_housekeeping(struct cgroup *cgrp)
{
 struct cgroup_subsys *ss;
 int ssid;

 INIT_LIST_HEAD(&cgrp->self.sibling);
 INIT_LIST_HEAD(&cgrp->self.children);
 INIT_LIST_HEAD(&cgrp->cset_links);
 INIT_LIST_HEAD(&cgrp->pidlists);
 mutex_init(&cgrp->pidlist_mutex);
 cgrp->self.cgroup = cgrp;
 cgrp->self.flags |= CSS_ONLINE;

 for_each_subsys(ss, ssid)
  INIT_LIST_HEAD(&cgrp->e_csets[ssid]);

 init_waitqueue_head(&cgrp->offline_waitq);
 INIT_WORK(&cgrp->release_agent_work, cgroup_release_agent);
}

static void init_cgroup_root(struct cgroup_root *root,
        struct cgroup_sb_opts *opts)
{
 struct cgroup *cgrp = &root->cgrp;

 INIT_LIST_HEAD(&root->root_list);
 atomic_set(&root->nr_cgrps, 1);
 cgrp->root = root;
 init_cgroup_housekeeping(cgrp);
 idr_init(&root->cgroup_idr);

 root->flags = opts->flags;
 if (opts->release_agent)
  strcpy(root->release_agent_path, opts->release_agent);
 if (opts->name)
  strcpy(root->name, opts->name);
 if (opts->cpuset_clone_children)
  set_bit(CGRP_CPUSET_CLONE_CHILDREN, &root->cgrp.flags);
}

static int cgroup_setup_root(struct cgroup_root *root, u16 ss_mask)
{
 LIST_HEAD(tmp_links);
 struct cgroup *root_cgrp = &root->cgrp;
 struct css_set *cset;
 int i, ret;

 lockdep_assert_held(&cgroup_mutex);

 ret = cgroup_idr_alloc(&root->cgroup_idr, root_cgrp, 1, 2, GFP_KERNEL);
 if (ret < 0)
  goto out;
 root_cgrp->id = ret;
 root_cgrp->ancestor_ids[0] = ret;

 ret = percpu_ref_init(&root_cgrp->self.refcnt, css_release, 0,
         GFP_KERNEL);
 if (ret)
  goto out;
 ret = allocate_cgrp_cset_links(2 * css_set_count, &tmp_links);
 if (ret)
  goto cancel_ref;

 ret = cgroup_init_root_id(root);
 if (ret)
  goto cancel_ref;

 root->kf_root = kernfs_create_root(&cgroup_kf_syscall_ops,
        KERNFS_ROOT_CREATE_DEACTIVATED,
        root_cgrp);
 if (IS_ERR(root->kf_root)) {
  ret = PTR_ERR(root->kf_root);
  goto exit_root_id;
 }
 root_cgrp->kn = root->kf_root->kn;

 ret = css_populate_dir(&root_cgrp->self);
 if (ret)
  goto destroy_root;

 ret = rebind_subsystems(root, ss_mask);
 if (ret)
  goto destroy_root;






 list_add(&root->root_list, &cgroup_roots);
 cgroup_root_count++;





 spin_lock_irq(&css_set_lock);
 hash_for_each(css_set_table, i, cset, hlist) {
  link_css_set(&tmp_links, cset, root_cgrp);
  if (css_set_populated(cset))
   cgroup_update_populated(root_cgrp, true);
 }
 spin_unlock_irq(&css_set_lock);

 BUG_ON(!list_empty(&root_cgrp->self.children));
 BUG_ON(atomic_read(&root->nr_cgrps) != 1);

 kernfs_activate(root_cgrp->kn);
 ret = 0;
 goto out;

destroy_root:
 kernfs_destroy_root(root->kf_root);
 root->kf_root = NULL;
exit_root_id:
 cgroup_exit_root_id(root);
cancel_ref:
 percpu_ref_exit(&root_cgrp->self.refcnt);
out:
 free_cgrp_cset_links(&tmp_links);
 return ret;
}

static struct dentry *cgroup_mount(struct file_system_type *fs_type,
    int flags, const char *unused_dev_name,
    void *data)
{
 bool is_v2 = fs_type == &cgroup2_fs_type;
 struct super_block *pinned_sb = NULL;
 struct cgroup_namespace *ns = current->nsproxy->cgroup_ns;
 struct cgroup_subsys *ss;
 struct cgroup_root *root;
 struct cgroup_sb_opts opts;
 struct dentry *dentry;
 int ret;
 int i;
 bool new_sb;

 get_cgroup_ns(ns);


 if (!ns_capable(ns->user_ns, CAP_SYS_ADMIN)) {
  put_cgroup_ns(ns);
  return ERR_PTR(-EPERM);
 }





 if (!use_task_css_set_links)
  cgroup_enable_task_cg_lists();

 if (is_v2) {
  if (data) {
   pr_err("cgroup2: unknown option \"%s\"\n", (char *)data);
   put_cgroup_ns(ns);
   return ERR_PTR(-EINVAL);
  }
  cgrp_dfl_visible = true;
  root = &cgrp_dfl_root;
  cgroup_get(&root->cgrp);
  goto out_mount;
 }

 cgroup_lock_and_drain_offline(&cgrp_dfl_root.cgrp);


 ret = parse_cgroupfs_options(data, &opts);
 if (ret)
  goto out_unlock;
 for_each_subsys(ss, i) {
  if (!(opts.subsys_mask & (1 << i)) ||
      ss->root == &cgrp_dfl_root)
   continue;

  if (!percpu_ref_tryget_live(&ss->root->cgrp.self.refcnt)) {
   mutex_unlock(&cgroup_mutex);
   msleep(10);
   ret = restart_syscall();
   goto out_free;
  }
  cgroup_put(&ss->root->cgrp);
 }

 for_each_root(root) {
  bool name_match = false;

  if (root == &cgrp_dfl_root)
   continue;






  if (opts.name) {
   if (strcmp(opts.name, root->name))
    continue;
   name_match = true;
  }





  if ((opts.subsys_mask || opts.none) &&
      (opts.subsys_mask != root->subsys_mask)) {
   if (!name_match)
    continue;
   ret = -EBUSY;
   goto out_unlock;
  }

  if (root->flags ^ opts.flags)
   pr_warn("new mount options do not match the existing superblock, will be ignored\n");
  pinned_sb = kernfs_pin_sb(root->kf_root, NULL);
  if (IS_ERR(pinned_sb) ||
      !percpu_ref_tryget_live(&root->cgrp.self.refcnt)) {
   mutex_unlock(&cgroup_mutex);
   if (!IS_ERR_OR_NULL(pinned_sb))
    deactivate_super(pinned_sb);
   msleep(10);
   ret = restart_syscall();
   goto out_free;
  }

  ret = 0;
  goto out_unlock;
 }






 if (!opts.subsys_mask && !opts.none) {
  ret = -EINVAL;
  goto out_unlock;
 }






 if (!opts.none && !capable(CAP_SYS_ADMIN)) {
  ret = -EPERM;
  goto out_unlock;
 }

 root = kzalloc(sizeof(*root), GFP_KERNEL);
 if (!root) {
  ret = -ENOMEM;
  goto out_unlock;
 }

 init_cgroup_root(root, &opts);

 ret = cgroup_setup_root(root, opts.subsys_mask);
 if (ret)
  cgroup_free_root(root);

out_unlock:
 mutex_unlock(&cgroup_mutex);
out_free:
 kfree(opts.release_agent);
 kfree(opts.name);

 if (ret) {
  put_cgroup_ns(ns);
  return ERR_PTR(ret);
 }
out_mount:
 dentry = kernfs_mount(fs_type, flags, root->kf_root,
         is_v2 ? CGROUP2_SUPER_MAGIC : CGROUP_SUPER_MAGIC,
         &new_sb);






 if (!IS_ERR(dentry) && ns != &init_cgroup_ns) {
  struct dentry *nsdentry;
  struct cgroup *cgrp;

  mutex_lock(&cgroup_mutex);
  spin_lock_irq(&css_set_lock);

  cgrp = cset_cgroup_from_root(ns->root_cset, root);

  spin_unlock_irq(&css_set_lock);
  mutex_unlock(&cgroup_mutex);

  nsdentry = kernfs_node_dentry(cgrp->kn, dentry->d_sb);
  dput(dentry);
  dentry = nsdentry;
 }

 if (IS_ERR(dentry) || !new_sb)
  cgroup_put(&root->cgrp);





 if (pinned_sb) {
  WARN_ON(new_sb);
  deactivate_super(pinned_sb);
 }

 put_cgroup_ns(ns);
 return dentry;
}

static void cgroup_kill_sb(struct super_block *sb)
{
 struct kernfs_root *kf_root = kernfs_root_from_sb(sb);
 struct cgroup_root *root = cgroup_root_from_kf(kf_root);
 if (!list_empty(&root->cgrp.self.children) ||
     root == &cgrp_dfl_root)
  cgroup_put(&root->cgrp);
 else
  percpu_ref_kill(&root->cgrp.self.refcnt);

 kernfs_kill_sb(sb);
}

static struct file_system_type cgroup_fs_type = {
 .name = "cgroup",
 .mount = cgroup_mount,
 .kill_sb = cgroup_kill_sb,
 .fs_flags = FS_USERNS_MOUNT,
};

static struct file_system_type cgroup2_fs_type = {
 .name = "cgroup2",
 .mount = cgroup_mount,
 .kill_sb = cgroup_kill_sb,
 .fs_flags = FS_USERNS_MOUNT,
};

static char *cgroup_path_ns_locked(struct cgroup *cgrp, char *buf, size_t buflen,
       struct cgroup_namespace *ns)
{
 struct cgroup *root = cset_cgroup_from_root(ns->root_cset, cgrp->root);
 int ret;

 ret = kernfs_path_from_node(cgrp->kn, root->kn, buf, buflen);
 if (ret < 0 || ret >= buflen)
  return NULL;
 return buf;
}

char *cgroup_path_ns(struct cgroup *cgrp, char *buf, size_t buflen,
       struct cgroup_namespace *ns)
{
 char *ret;

 mutex_lock(&cgroup_mutex);
 spin_lock_irq(&css_set_lock);

 ret = cgroup_path_ns_locked(cgrp, buf, buflen, ns);

 spin_unlock_irq(&css_set_lock);
 mutex_unlock(&cgroup_mutex);

 return ret;
}
EXPORT_SYMBOL_GPL(cgroup_path_ns);
char *task_cgroup_path(struct task_struct *task, char *buf, size_t buflen)
{
 struct cgroup_root *root;
 struct cgroup *cgrp;
 int hierarchy_id = 1;
 char *path = NULL;

 mutex_lock(&cgroup_mutex);
 spin_lock_irq(&css_set_lock);

 root = idr_get_next(&cgroup_hierarchy_idr, &hierarchy_id);

 if (root) {
  cgrp = task_cgroup_from_root(task, root);
  path = cgroup_path_ns_locked(cgrp, buf, buflen, &init_cgroup_ns);
 } else {

  if (strlcpy(buf, "/", buflen) < buflen)
   path = buf;
 }

 spin_unlock_irq(&css_set_lock);
 mutex_unlock(&cgroup_mutex);
 return path;
}
EXPORT_SYMBOL_GPL(task_cgroup_path);


struct cgroup_taskset {

 struct list_head src_csets;
 struct list_head dst_csets;


 int ssid;
 struct list_head *csets;
 struct css_set *cur_cset;
 struct task_struct *cur_task;
};

 .src_csets = LIST_HEAD_INIT(tset.src_csets), \
 .dst_csets = LIST_HEAD_INIT(tset.dst_csets), \
 .csets = &tset.src_csets, \
}
static void cgroup_taskset_add(struct task_struct *task,
          struct cgroup_taskset *tset)
{
 struct css_set *cset;

 lockdep_assert_held(&css_set_lock);


 if (task->flags & PF_EXITING)
  return;


 if (list_empty(&task->cg_list))
  return;

 cset = task_css_set(task);
 if (!cset->mg_src_cgrp)
  return;

 list_move_tail(&task->cg_list, &cset->mg_tasks);
 if (list_empty(&cset->mg_node))
  list_add_tail(&cset->mg_node, &tset->src_csets);
 if (list_empty(&cset->mg_dst_cset->mg_node))
  list_move_tail(&cset->mg_dst_cset->mg_node,
          &tset->dst_csets);
}
struct task_struct *cgroup_taskset_first(struct cgroup_taskset *tset,
      struct cgroup_subsys_state **dst_cssp)
{
 tset->cur_cset = list_first_entry(tset->csets, struct css_set, mg_node);
 tset->cur_task = NULL;

 return cgroup_taskset_next(tset, dst_cssp);
}
struct task_struct *cgroup_taskset_next(struct cgroup_taskset *tset,
     struct cgroup_subsys_state **dst_cssp)
{
 struct css_set *cset = tset->cur_cset;
 struct task_struct *task = tset->cur_task;

 while (&cset->mg_node != tset->csets) {
  if (!task)
   task = list_first_entry(&cset->mg_tasks,
      struct task_struct, cg_list);
  else
   task = list_next_entry(task, cg_list);

  if (&task->cg_list != &cset->mg_tasks) {
   tset->cur_cset = cset;
   tset->cur_task = task;







   if (cset->mg_dst_cset)
    *dst_cssp = cset->mg_dst_cset->subsys[tset->ssid];
   else
    *dst_cssp = cset->subsys[tset->ssid];

   return task;
  }

  cset = list_next_entry(cset, mg_node);
  task = NULL;
 }

 return NULL;
}
static int cgroup_taskset_migrate(struct cgroup_taskset *tset,
      struct cgroup_root *root)
{
 struct cgroup_subsys *ss;
 struct task_struct *task, *tmp_task;
 struct css_set *cset, *tmp_cset;
 int ssid, failed_ssid, ret;


 if (list_empty(&tset->src_csets))
  return 0;


 do_each_subsys_mask(ss, ssid, root->subsys_mask) {
  if (ss->can_attach) {
   tset->ssid = ssid;
   ret = ss->can_attach(tset);
   if (ret) {
    failed_ssid = ssid;
    goto out_cancel_attach;
   }
  }
 } while_each_subsys_mask();






 spin_lock_irq(&css_set_lock);
 list_for_each_entry(cset, &tset->src_csets, mg_node) {
  list_for_each_entry_safe(task, tmp_task, &cset->mg_tasks, cg_list) {
   struct css_set *from_cset = task_css_set(task);
   struct css_set *to_cset = cset->mg_dst_cset;

   get_css_set(to_cset);
   css_set_move_task(task, from_cset, to_cset, true);
   put_css_set_locked(from_cset);
  }
 }
 spin_unlock_irq(&css_set_lock);






 tset->csets = &tset->dst_csets;

 do_each_subsys_mask(ss, ssid, root->subsys_mask) {
  if (ss->attach) {
   tset->ssid = ssid;
   ss->attach(tset);
  }
 } while_each_subsys_mask();

 ret = 0;
 goto out_release_tset;

out_cancel_attach:
 do_each_subsys_mask(ss, ssid, root->subsys_mask) {
  if (ssid == failed_ssid)
   break;
  if (ss->cancel_attach) {
   tset->ssid = ssid;
   ss->cancel_attach(tset);
  }
 } while_each_subsys_mask();
out_release_tset:
 spin_lock_irq(&css_set_lock);
 list_splice_init(&tset->dst_csets, &tset->src_csets);
 list_for_each_entry_safe(cset, tmp_cset, &tset->src_csets, mg_node) {
  list_splice_tail_init(&cset->mg_tasks, &cset->tasks);
  list_del_init(&cset->mg_node);
 }
 spin_unlock_irq(&css_set_lock);
 return ret;
}
static bool cgroup_may_migrate_to(struct cgroup *dst_cgrp)
{
 return !cgroup_on_dfl(dst_cgrp) || !cgroup_parent(dst_cgrp) ||
  !dst_cgrp->subtree_control;
}
static void cgroup_migrate_finish(struct list_head *preloaded_csets)
{
 struct css_set *cset, *tmp_cset;

 lockdep_assert_held(&cgroup_mutex);

 spin_lock_irq(&css_set_lock);
 list_for_each_entry_safe(cset, tmp_cset, preloaded_csets, mg_preload_node) {
  cset->mg_src_cgrp = NULL;
  cset->mg_dst_cgrp = NULL;
  cset->mg_dst_cset = NULL;
  list_del_init(&cset->mg_preload_node);
  put_css_set_locked(cset);
 }
 spin_unlock_irq(&css_set_lock);
}
static void cgroup_migrate_add_src(struct css_set *src_cset,
       struct cgroup *dst_cgrp,
       struct list_head *preloaded_csets)
{
 struct cgroup *src_cgrp;

 lockdep_assert_held(&cgroup_mutex);
 lockdep_assert_held(&css_set_lock);






 if (src_cset->dead)
  return;

 src_cgrp = cset_cgroup_from_root(src_cset, dst_cgrp->root);

 if (!list_empty(&src_cset->mg_preload_node))
  return;

 WARN_ON(src_cset->mg_src_cgrp);
 WARN_ON(src_cset->mg_dst_cgrp);
 WARN_ON(!list_empty(&src_cset->mg_tasks));
 WARN_ON(!list_empty(&src_cset->mg_node));

 src_cset->mg_src_cgrp = src_cgrp;
 src_cset->mg_dst_cgrp = dst_cgrp;
 get_css_set(src_cset);
 list_add(&src_cset->mg_preload_node, preloaded_csets);
}
static int cgroup_migrate_prepare_dst(struct list_head *preloaded_csets)
{
 LIST_HEAD(csets);
 struct css_set *src_cset, *tmp_cset;

 lockdep_assert_held(&cgroup_mutex);


 list_for_each_entry_safe(src_cset, tmp_cset, preloaded_csets, mg_preload_node) {
  struct css_set *dst_cset;

  dst_cset = find_css_set(src_cset, src_cset->mg_dst_cgrp);
  if (!dst_cset)
   goto err;

  WARN_ON_ONCE(src_cset->mg_dst_cset || dst_cset->mg_dst_cset);






  if (src_cset == dst_cset) {
   src_cset->mg_src_cgrp = NULL;
   src_cset->mg_dst_cgrp = NULL;
   list_del_init(&src_cset->mg_preload_node);
   put_css_set(src_cset);
   put_css_set(dst_cset);
   continue;
  }

  src_cset->mg_dst_cset = dst_cset;

  if (list_empty(&dst_cset->mg_preload_node))
   list_add(&dst_cset->mg_preload_node, &csets);
  else
   put_css_set(dst_cset);
 }

 list_splice_tail(&csets, preloaded_csets);
 return 0;
err:
 cgroup_migrate_finish(&csets);
 return -ENOMEM;
}
static int cgroup_migrate(struct task_struct *leader, bool threadgroup,
     struct cgroup_root *root)
{
 struct cgroup_taskset tset = CGROUP_TASKSET_INIT(tset);
 struct task_struct *task;






 spin_lock_irq(&css_set_lock);
 rcu_read_lock();
 task = leader;
 do {
  cgroup_taskset_add(task, &tset);
  if (!threadgroup)
   break;
 } while_each_thread(leader, task);
 rcu_read_unlock();
 spin_unlock_irq(&css_set_lock);

 return cgroup_taskset_migrate(&tset, root);
}
static int cgroup_attach_task(struct cgroup *dst_cgrp,
         struct task_struct *leader, bool threadgroup)
{
 LIST_HEAD(preloaded_csets);
 struct task_struct *task;
 int ret;

 if (!cgroup_may_migrate_to(dst_cgrp))
  return -EBUSY;


 spin_lock_irq(&css_set_lock);
 rcu_read_lock();
 task = leader;
 do {
  cgroup_migrate_add_src(task_css_set(task), dst_cgrp,
           &preloaded_csets);
  if (!threadgroup)
   break;
 } while_each_thread(leader, task);
 rcu_read_unlock();
 spin_unlock_irq(&css_set_lock);


 ret = cgroup_migrate_prepare_dst(&preloaded_csets);
 if (!ret)
  ret = cgroup_migrate(leader, threadgroup, dst_cgrp->root);

 cgroup_migrate_finish(&preloaded_csets);
 return ret;
}

static int cgroup_procs_write_permission(struct task_struct *task,
      struct cgroup *dst_cgrp,
      struct kernfs_open_file *of)
{
 const struct cred *cred = current_cred();
 const struct cred *tcred = get_task_cred(task);
 int ret = 0;





 if (!uid_eq(cred->euid, GLOBAL_ROOT_UID) &&
     !uid_eq(cred->euid, tcred->uid) &&
     !uid_eq(cred->euid, tcred->suid))
  ret = -EACCES;

 if (!ret && cgroup_on_dfl(dst_cgrp)) {
  struct super_block *sb = of->file->f_path.dentry->d_sb;
  struct cgroup *cgrp;
  struct inode *inode;

  spin_lock_irq(&css_set_lock);
  cgrp = task_cgroup_from_root(task, &cgrp_dfl_root);
  spin_unlock_irq(&css_set_lock);

  while (!cgroup_is_descendant(dst_cgrp, cgrp))
   cgrp = cgroup_parent(cgrp);

  ret = -ENOMEM;
  inode = kernfs_get_inode(sb, cgrp->procs_file.kn);
  if (inode) {
   ret = inode_permission(inode, MAY_WRITE);
   iput(inode);
  }
 }

 put_cred(tcred);
 return ret;
}






static ssize_t __cgroup_procs_write(struct kernfs_open_file *of, char *buf,
        size_t nbytes, loff_t off, bool threadgroup)
{
 struct task_struct *tsk;
 struct cgroup_subsys *ss;
 struct cgroup *cgrp;
 pid_t pid;
 int ssid, ret;

 if (kstrtoint(strstrip(buf), 0, &pid) || pid < 0)
  return -EINVAL;

 cgrp = cgroup_kn_lock_live(of->kn, false);
 if (!cgrp)
  return -ENODEV;

 percpu_down_write(&cgroup_threadgroup_rwsem);
 rcu_read_lock();
 if (pid) {
  tsk = find_task_by_vpid(pid);
  if (!tsk) {
   ret = -ESRCH;
   goto out_unlock_rcu;
  }
 } else {
  tsk = current;
 }

 if (threadgroup)
  tsk = tsk->group_leader;






 if (tsk == kthreadd_task || (tsk->flags & PF_NO_SETAFFINITY)) {
  ret = -EINVAL;
  goto out_unlock_rcu;
 }

 get_task_struct(tsk);
 rcu_read_unlock();

 ret = cgroup_procs_write_permission(tsk, cgrp, of);
 if (!ret)
  ret = cgroup_attach_task(cgrp, tsk, threadgroup);

 put_task_struct(tsk);
 goto out_unlock_threadgroup;

out_unlock_rcu:
 rcu_read_unlock();
out_unlock_threadgroup:
 percpu_up_write(&cgroup_threadgroup_rwsem);
 for_each_subsys(ss, ssid)
  if (ss->post_attach)
   ss->post_attach();
 cgroup_kn_unlock(of->kn);
 return ret ?: nbytes;
}






int cgroup_attach_task_all(struct task_struct *from, struct task_struct *tsk)
{
 struct cgroup_root *root;
 int retval = 0;

 mutex_lock(&cgroup_mutex);
 for_each_root(root) {
  struct cgroup *from_cgrp;

  if (root == &cgrp_dfl_root)
   continue;

  spin_lock_irq(&css_set_lock);
  from_cgrp = task_cgroup_from_root(from, root);
  spin_unlock_irq(&css_set_lock);

  retval = cgroup_attach_task(from_cgrp, tsk, false);
  if (retval)
   break;
 }
 mutex_unlock(&cgroup_mutex);

 return retval;
}
EXPORT_SYMBOL_GPL(cgroup_attach_task_all);

static ssize_t cgroup_tasks_write(struct kernfs_open_file *of,
      char *buf, size_t nbytes, loff_t off)
{
 return __cgroup_procs_write(of, buf, nbytes, off, false);
}

static ssize_t cgroup_procs_write(struct kernfs_open_file *of,
      char *buf, size_t nbytes, loff_t off)
{
 return __cgroup_procs_write(of, buf, nbytes, off, true);
}

static ssize_t cgroup_release_agent_write(struct kernfs_open_file *of,
       char *buf, size_t nbytes, loff_t off)
{
 struct cgroup *cgrp;

 BUILD_BUG_ON(sizeof(cgrp->root->release_agent_path) < PATH_MAX);

 cgrp = cgroup_kn_lock_live(of->kn, false);
 if (!cgrp)
  return -ENODEV;
 spin_lock(&release_agent_path_lock);
 strlcpy(cgrp->root->release_agent_path, strstrip(buf),
  sizeof(cgrp->root->release_agent_path));
 spin_unlock(&release_agent_path_lock);
 cgroup_kn_unlock(of->kn);
 return nbytes;
}

static int cgroup_release_agent_show(struct seq_file *seq, void *v)
{
 struct cgroup *cgrp = seq_css(seq)->cgroup;

 spin_lock(&release_agent_path_lock);
 seq_puts(seq, cgrp->root->release_agent_path);
 spin_unlock(&release_agent_path_lock);
 seq_putc(seq, '\n');
 return 0;
}

static int cgroup_sane_behavior_show(struct seq_file *seq, void *v)
{
 seq_puts(seq, "0\n");
 return 0;
}

static void cgroup_print_ss_mask(struct seq_file *seq, u16 ss_mask)
{
 struct cgroup_subsys *ss;
 bool printed = false;
 int ssid;

 do_each_subsys_mask(ss, ssid, ss_mask) {
  if (printed)
   seq_putc(seq, ' ');
  seq_printf(seq, "%s", ss->name);
  printed = true;
 } while_each_subsys_mask();
 if (printed)
  seq_putc(seq, '\n');
}


static int cgroup_controllers_show(struct seq_file *seq, void *v)
{
 struct cgroup *cgrp = seq_css(seq)->cgroup;

 cgroup_print_ss_mask(seq, cgroup_control(cgrp));
 return 0;
}


static int cgroup_subtree_control_show(struct seq_file *seq, void *v)
{
 struct cgroup *cgrp = seq_css(seq)->cgroup;

 cgroup_print_ss_mask(seq, cgrp->subtree_control);
 return 0;
}
static int cgroup_update_dfl_csses(struct cgroup *cgrp)
{
 LIST_HEAD(preloaded_csets);
 struct cgroup_taskset tset = CGROUP_TASKSET_INIT(tset);
 struct cgroup_subsys_state *d_css;
 struct cgroup *dsct;
 struct css_set *src_cset;
 int ret;

 lockdep_assert_held(&cgroup_mutex);

 percpu_down_write(&cgroup_threadgroup_rwsem);


 spin_lock_irq(&css_set_lock);
 cgroup_for_each_live_descendant_pre(dsct, d_css, cgrp) {
  struct cgrp_cset_link *link;

  list_for_each_entry(link, &dsct->cset_links, cset_link)
   cgroup_migrate_add_src(link->cset, dsct,
            &preloaded_csets);
 }
 spin_unlock_irq(&css_set_lock);


 ret = cgroup_migrate_prepare_dst(&preloaded_csets);
 if (ret)
  goto out_finish;

 spin_lock_irq(&css_set_lock);
 list_for_each_entry(src_cset, &preloaded_csets, mg_preload_node) {
  struct task_struct *task, *ntask;


  if (!src_cset->mg_src_cgrp)
   break;


  list_for_each_entry_safe(task, ntask, &src_cset->tasks, cg_list)
   cgroup_taskset_add(task, &tset);
 }
 spin_unlock_irq(&css_set_lock);

 ret = cgroup_taskset_migrate(&tset, cgrp->root);
out_finish:
 cgroup_migrate_finish(&preloaded_csets);
 percpu_up_write(&cgroup_threadgroup_rwsem);
 return ret;
}
static void cgroup_lock_and_drain_offline(struct cgroup *cgrp)
 __acquires(&cgroup_mutex)
{
 struct cgroup *dsct;
 struct cgroup_subsys_state *d_css;
 struct cgroup_subsys *ss;
 int ssid;

restart:
 mutex_lock(&cgroup_mutex);

 cgroup_for_each_live_descendant_post(dsct, d_css, cgrp) {
  for_each_subsys(ss, ssid) {
   struct cgroup_subsys_state *css = cgroup_css(dsct, ss);
   DEFINE_WAIT(wait);

   if (!css || !percpu_ref_is_dying(&css->refcnt))
    continue;

   cgroup_get(dsct);
   prepare_to_wait(&dsct->offline_waitq, &wait,
     TASK_UNINTERRUPTIBLE);

   mutex_unlock(&cgroup_mutex);
   schedule();
   finish_wait(&dsct->offline_waitq, &wait);

   cgroup_put(dsct);
   goto restart;
  }
 }
}
static void cgroup_save_control(struct cgroup *cgrp)
{
 struct cgroup *dsct;
 struct cgroup_subsys_state *d_css;

 cgroup_for_each_live_descendant_pre(dsct, d_css, cgrp) {
  dsct->old_subtree_control = dsct->subtree_control;
  dsct->old_subtree_ss_mask = dsct->subtree_ss_mask;
 }
}
static void cgroup_propagate_control(struct cgroup *cgrp)
{
 struct cgroup *dsct;
 struct cgroup_subsys_state *d_css;

 cgroup_for_each_live_descendant_pre(dsct, d_css, cgrp) {
  dsct->subtree_control &= cgroup_control(dsct);
  dsct->subtree_ss_mask =
   cgroup_calc_subtree_ss_mask(dsct->subtree_control,
          cgroup_ss_mask(dsct));
 }
}
static void cgroup_restore_control(struct cgroup *cgrp)
{
 struct cgroup *dsct;
 struct cgroup_subsys_state *d_css;

 cgroup_for_each_live_descendant_post(dsct, d_css, cgrp) {
  dsct->subtree_control = dsct->old_subtree_control;
  dsct->subtree_ss_mask = dsct->old_subtree_ss_mask;
 }
}

static bool css_visible(struct cgroup_subsys_state *css)
{
 struct cgroup_subsys *ss = css->ss;
 struct cgroup *cgrp = css->cgroup;

 if (cgroup_control(cgrp) & (1 << ss->id))
  return true;
 if (!(cgroup_ss_mask(cgrp) & (1 << ss->id)))
  return false;
 return cgroup_on_dfl(cgrp) && ss->implicit_on_dfl;
}
static int cgroup_apply_control_enable(struct cgroup *cgrp)
{
 struct cgroup *dsct;
 struct cgroup_subsys_state *d_css;
 struct cgroup_subsys *ss;
 int ssid, ret;

 cgroup_for_each_live_descendant_pre(dsct, d_css, cgrp) {
  for_each_subsys(ss, ssid) {
   struct cgroup_subsys_state *css = cgroup_css(dsct, ss);

   WARN_ON_ONCE(css && percpu_ref_is_dying(&css->refcnt));

   if (!(cgroup_ss_mask(dsct) & (1 << ss->id)))
    continue;

   if (!css) {
    css = css_create(dsct, ss);
    if (IS_ERR(css))
     return PTR_ERR(css);
   }

   if (css_visible(css)) {
    ret = css_populate_dir(css);
    if (ret)
     return ret;
   }
  }
 }

 return 0;
}
static void cgroup_apply_control_disable(struct cgroup *cgrp)
{
 struct cgroup *dsct;
 struct cgroup_subsys_state *d_css;
 struct cgroup_subsys *ss;
 int ssid;

 cgroup_for_each_live_descendant_post(dsct, d_css, cgrp) {
  for_each_subsys(ss, ssid) {
   struct cgroup_subsys_state *css = cgroup_css(dsct, ss);

   WARN_ON_ONCE(css && percpu_ref_is_dying(&css->refcnt));

   if (!css)
    continue;

   if (css->parent &&
       !(cgroup_ss_mask(dsct) & (1 << ss->id))) {
    kill_css(css);
   } else if (!css_visible(css)) {
    css_clear_dir(css);
    if (ss->css_reset)
     ss->css_reset(css);
   }
  }
 }
}
static int cgroup_apply_control(struct cgroup *cgrp)
{
 int ret;

 cgroup_propagate_control(cgrp);

 ret = cgroup_apply_control_enable(cgrp);
 if (ret)
  return ret;






 ret = cgroup_update_dfl_csses(cgrp);
 if (ret)
  return ret;

 return 0;
}
static void cgroup_finalize_control(struct cgroup *cgrp, int ret)
{
 if (ret) {
  cgroup_restore_control(cgrp);
  cgroup_propagate_control(cgrp);
 }

 cgroup_apply_control_disable(cgrp);
}


static ssize_t cgroup_subtree_control_write(struct kernfs_open_file *of,
         char *buf, size_t nbytes,
         loff_t off)
{
 u16 enable = 0, disable = 0;
 struct cgroup *cgrp, *child;
 struct cgroup_subsys *ss;
 char *tok;
 int ssid, ret;





 buf = strstrip(buf);
 while ((tok = strsep(&buf, " "))) {
  if (tok[0] == '\0')
   continue;
  do_each_subsys_mask(ss, ssid, ~cgrp_dfl_inhibit_ss_mask) {
   if (!cgroup_ssid_enabled(ssid) ||
       strcmp(tok + 1, ss->name))
    continue;

   if (*tok == '+') {
    enable |= 1 << ssid;
    disable &= ~(1 << ssid);
   } else if (*tok == '-') {
    disable |= 1 << ssid;
    enable &= ~(1 << ssid);
   } else {
    return -EINVAL;
   }
   break;
  } while_each_subsys_mask();
  if (ssid == CGROUP_SUBSYS_COUNT)
   return -EINVAL;
 }

 cgrp = cgroup_kn_lock_live(of->kn, true);
 if (!cgrp)
  return -ENODEV;

 for_each_subsys(ss, ssid) {
  if (enable & (1 << ssid)) {
   if (cgrp->subtree_control & (1 << ssid)) {
    enable &= ~(1 << ssid);
    continue;
   }

   if (!(cgroup_control(cgrp) & (1 << ssid))) {
    ret = -ENOENT;
    goto out_unlock;
   }
  } else if (disable & (1 << ssid)) {
   if (!(cgrp->subtree_control & (1 << ssid))) {
    disable &= ~(1 << ssid);
    continue;
   }


   cgroup_for_each_live_child(child, cgrp) {
    if (child->subtree_control & (1 << ssid)) {
     ret = -EBUSY;
     goto out_unlock;
    }
   }
  }
 }

 if (!enable && !disable) {
  ret = 0;
  goto out_unlock;
 }





 if (enable && cgroup_parent(cgrp) && !list_empty(&cgrp->cset_links)) {
  ret = -EBUSY;
  goto out_unlock;
 }


 cgroup_save_control(cgrp);

 cgrp->subtree_control |= enable;
 cgrp->subtree_control &= ~disable;

 ret = cgroup_apply_control(cgrp);

 cgroup_finalize_control(cgrp, ret);

 kernfs_activate(cgrp->kn);
 ret = 0;
out_unlock:
 cgroup_kn_unlock(of->kn);
 return ret ?: nbytes;
}

static int cgroup_events_show(struct seq_file *seq, void *v)
{
 seq_printf(seq, "populated %d\n",
     cgroup_is_populated(seq_css(seq)->cgroup));
 return 0;
}

static ssize_t cgroup_file_write(struct kernfs_open_file *of, char *buf,
     size_t nbytes, loff_t off)
{
 struct cgroup *cgrp = of->kn->parent->priv;
 struct cftype *cft = of->kn->priv;
 struct cgroup_subsys_state *css;
 int ret;

 if (cft->write)
  return cft->write(of, buf, nbytes, off);







 rcu_read_lock();
 css = cgroup_css(cgrp, cft->ss);
 rcu_read_unlock();

 if (cft->write_u64) {
  unsigned long long v;
  ret = kstrtoull(buf, 0, &v);
  if (!ret)
   ret = cft->write_u64(css, cft, v);
 } else if (cft->write_s64) {
  long long v;
  ret = kstrtoll(buf, 0, &v);
  if (!ret)
   ret = cft->write_s64(css, cft, v);
 } else {
  ret = -EINVAL;
 }

 return ret ?: nbytes;
}

static void *cgroup_seqfile_start(struct seq_file *seq, loff_t *ppos)
{
 return seq_cft(seq)->seq_start(seq, ppos);
}

static void *cgroup_seqfile_next(struct seq_file *seq, void *v, loff_t *ppos)
{
 return seq_cft(seq)->seq_next(seq, v, ppos);
}

static void cgroup_seqfile_stop(struct seq_file *seq, void *v)
{
 seq_cft(seq)->seq_stop(seq, v);
}

static int cgroup_seqfile_show(struct seq_file *m, void *arg)
{
 struct cftype *cft = seq_cft(m);
 struct cgroup_subsys_state *css = seq_css(m);

 if (cft->seq_show)
  return cft->seq_show(m, arg);

 if (cft->read_u64)
  seq_printf(m, "%llu\n", cft->read_u64(css, cft));
 else if (cft->read_s64)
  seq_printf(m, "%lld\n", cft->read_s64(css, cft));
 else
  return -EINVAL;
 return 0;
}

static struct kernfs_ops cgroup_kf_single_ops = {
 .atomic_write_len = PAGE_SIZE,
 .write = cgroup_file_write,
 .seq_show = cgroup_seqfile_show,
};

static struct kernfs_ops cgroup_kf_ops = {
 .atomic_write_len = PAGE_SIZE,
 .write = cgroup_file_write,
 .seq_start = cgroup_seqfile_start,
 .seq_next = cgroup_seqfile_next,
 .seq_stop = cgroup_seqfile_stop,
 .seq_show = cgroup_seqfile_show,
};




static int cgroup_rename(struct kernfs_node *kn, struct kernfs_node *new_parent,
    const char *new_name_str)
{
 struct cgroup *cgrp = kn->priv;
 int ret;

 if (kernfs_type(kn) != KERNFS_DIR)
  return -ENOTDIR;
 if (kn->parent != new_parent)
  return -EIO;





 if (cgroup_on_dfl(cgrp))
  return -EPERM;






 kernfs_break_active_protection(new_parent);
 kernfs_break_active_protection(kn);

 mutex_lock(&cgroup_mutex);

 ret = kernfs_rename(kn, new_parent, new_name_str);

 mutex_unlock(&cgroup_mutex);

 kernfs_unbreak_active_protection(kn);
 kernfs_unbreak_active_protection(new_parent);
 return ret;
}


static int cgroup_kn_set_ugid(struct kernfs_node *kn)
{
 struct iattr iattr = { .ia_valid = ATTR_UID | ATTR_GID,
          .ia_uid = current_fsuid(),
          .ia_gid = current_fsgid(), };

 if (uid_eq(iattr.ia_uid, GLOBAL_ROOT_UID) &&
     gid_eq(iattr.ia_gid, GLOBAL_ROOT_GID))
  return 0;

 return kernfs_setattr(kn, &iattr);
}

static int cgroup_add_file(struct cgroup_subsys_state *css, struct cgroup *cgrp,
      struct cftype *cft)
{
 char name[CGROUP_FILE_NAME_MAX];
 struct kernfs_node *kn;
 struct lock_class_key *key = NULL;
 int ret;

 key = &cft->lockdep_key;
 kn = __kernfs_create_file(cgrp->kn, cgroup_file_name(cgrp, cft, name),
      cgroup_file_mode(cft), 0, cft->kf_ops, cft,
      NULL, key);
 if (IS_ERR(kn))
  return PTR_ERR(kn);

 ret = cgroup_kn_set_ugid(kn);
 if (ret) {
  kernfs_remove(kn);
  return ret;
 }

 if (cft->file_offset) {
  struct cgroup_file *cfile = (void *)css + cft->file_offset;

  spin_lock_irq(&cgroup_file_kn_lock);
  cfile->kn = kn;
  spin_unlock_irq(&cgroup_file_kn_lock);
 }

 return 0;
}
static int cgroup_addrm_files(struct cgroup_subsys_state *css,
         struct cgroup *cgrp, struct cftype cfts[],
         bool is_add)
{
 struct cftype *cft, *cft_end = NULL;
 int ret = 0;

 lockdep_assert_held(&cgroup_mutex);

restart:
 for (cft = cfts; cft != cft_end && cft->name[0] != '\0'; cft++) {

  if ((cft->flags & __CFTYPE_ONLY_ON_DFL) && !cgroup_on_dfl(cgrp))
   continue;
  if ((cft->flags & __CFTYPE_NOT_ON_DFL) && cgroup_on_dfl(cgrp))
   continue;
  if ((cft->flags & CFTYPE_NOT_ON_ROOT) && !cgroup_parent(cgrp))
   continue;
  if ((cft->flags & CFTYPE_ONLY_ON_ROOT) && cgroup_parent(cgrp))
   continue;

  if (is_add) {
   ret = cgroup_add_file(css, cgrp, cft);
   if (ret) {
    pr_warn("%s: failed to add %s, err=%d\n",
     __func__, cft->name, ret);
    cft_end = cft;
    is_add = false;
    goto restart;
   }
  } else {
   cgroup_rm_file(cgrp, cft);
  }
 }
 return ret;
}

static int cgroup_apply_cftypes(struct cftype *cfts, bool is_add)
{
 LIST_HEAD(pending);
 struct cgroup_subsys *ss = cfts[0].ss;
 struct cgroup *root = &ss->root->cgrp;
 struct cgroup_subsys_state *css;
 int ret = 0;

 lockdep_assert_held(&cgroup_mutex);


 css_for_each_descendant_pre(css, cgroup_css(root, ss)) {
  struct cgroup *cgrp = css->cgroup;

  if (!(css->flags & CSS_VISIBLE))
   continue;

  ret = cgroup_addrm_files(css, cgrp, cfts, is_add);
  if (ret)
   break;
 }

 if (is_add && !ret)
  kernfs_activate(root->kn);
 return ret;
}

static void cgroup_exit_cftypes(struct cftype *cfts)
{
 struct cftype *cft;

 for (cft = cfts; cft->name[0] != '\0'; cft++) {

  if (cft->max_write_len && cft->max_write_len != PAGE_SIZE)
   kfree(cft->kf_ops);
  cft->kf_ops = NULL;
  cft->ss = NULL;


  cft->flags &= ~(__CFTYPE_ONLY_ON_DFL | __CFTYPE_NOT_ON_DFL);
 }
}

static int cgroup_init_cftypes(struct cgroup_subsys *ss, struct cftype *cfts)
{
 struct cftype *cft;

 for (cft = cfts; cft->name[0] != '\0'; cft++) {
  struct kernfs_ops *kf_ops;

  WARN_ON(cft->ss || cft->kf_ops);

  if (cft->seq_start)
   kf_ops = &cgroup_kf_ops;
  else
   kf_ops = &cgroup_kf_single_ops;





  if (cft->max_write_len && cft->max_write_len != PAGE_SIZE) {
   kf_ops = kmemdup(kf_ops, sizeof(*kf_ops), GFP_KERNEL);
   if (!kf_ops) {
    cgroup_exit_cftypes(cfts);
    return -ENOMEM;
   }
   kf_ops->atomic_write_len = cft->max_write_len;
  }

  cft->kf_ops = kf_ops;
  cft->ss = ss;
 }

 return 0;
}

static int cgroup_rm_cftypes_locked(struct cftype *cfts)
{
 lockdep_assert_held(&cgroup_mutex);

 if (!cfts || !cfts[0].ss)
  return -ENOENT;

 list_del(&cfts->node);
 cgroup_apply_cftypes(cfts, false);
 cgroup_exit_cftypes(cfts);
 return 0;
}
int cgroup_rm_cftypes(struct cftype *cfts)
{
 int ret;

 mutex_lock(&cgroup_mutex);
 ret = cgroup_rm_cftypes_locked(cfts);
 mutex_unlock(&cgroup_mutex);
 return ret;
}
static int cgroup_add_cftypes(struct cgroup_subsys *ss, struct cftype *cfts)
{
 int ret;

 if (!cgroup_ssid_enabled(ss->id))
  return 0;

 if (!cfts || cfts[0].name[0] == '\0')
  return 0;

 ret = cgroup_init_cftypes(ss, cfts);
 if (ret)
  return ret;

 mutex_lock(&cgroup_mutex);

 list_add_tail(&cfts->node, &ss->cfts);
 ret = cgroup_apply_cftypes(cfts, true);
 if (ret)
  cgroup_rm_cftypes_locked(cfts);

 mutex_unlock(&cgroup_mutex);
 return ret;
}
int cgroup_add_dfl_cftypes(struct cgroup_subsys *ss, struct cftype *cfts)
{
 struct cftype *cft;

 for (cft = cfts; cft && cft->name[0] != '\0'; cft++)
  cft->flags |= __CFTYPE_ONLY_ON_DFL;
 return cgroup_add_cftypes(ss, cfts);
}
int cgroup_add_legacy_cftypes(struct cgroup_subsys *ss, struct cftype *cfts)
{
 struct cftype *cft;

 for (cft = cfts; cft && cft->name[0] != '\0'; cft++)
  cft->flags |= __CFTYPE_NOT_ON_DFL;
 return cgroup_add_cftypes(ss, cfts);
}







void cgroup_file_notify(struct cgroup_file *cfile)
{
 unsigned long flags;

 spin_lock_irqsave(&cgroup_file_kn_lock, flags);
 if (cfile->kn)
  kernfs_notify(cfile->kn);
 spin_unlock_irqrestore(&cgroup_file_kn_lock, flags);
}







static int cgroup_task_count(const struct cgroup *cgrp)
{
 int count = 0;
 struct cgrp_cset_link *link;

 spin_lock_irq(&css_set_lock);
 list_for_each_entry(link, &cgrp->cset_links, cset_link)
  count += atomic_read(&link->cset->refcount);
 spin_unlock_irq(&css_set_lock);
 return count;
}
struct cgroup_subsys_state *css_next_child(struct cgroup_subsys_state *pos,
        struct cgroup_subsys_state *parent)
{
 struct cgroup_subsys_state *next;

 cgroup_assert_mutex_or_rcu_locked();
 if (!pos) {
  next = list_entry_rcu(parent->children.next, struct cgroup_subsys_state, sibling);
 } else if (likely(!(pos->flags & CSS_RELEASED))) {
  next = list_entry_rcu(pos->sibling.next, struct cgroup_subsys_state, sibling);
 } else {
  list_for_each_entry_rcu(next, &parent->children, sibling)
   if (next->serial_nr > pos->serial_nr)
    break;
 }





 if (&next->sibling != &parent->children)
  return next;
 return NULL;
}
struct cgroup_subsys_state *
css_next_descendant_pre(struct cgroup_subsys_state *pos,
   struct cgroup_subsys_state *root)
{
 struct cgroup_subsys_state *next;

 cgroup_assert_mutex_or_rcu_locked();


 if (!pos)
  return root;


 next = css_next_child(NULL, pos);
 if (next)
  return next;


 while (pos != root) {
  next = css_next_child(pos, pos->parent);
  if (next)
   return next;
  pos = pos->parent;
 }

 return NULL;
}
struct cgroup_subsys_state *
css_rightmost_descendant(struct cgroup_subsys_state *pos)
{
 struct cgroup_subsys_state *last, *tmp;

 cgroup_assert_mutex_or_rcu_locked();

 do {
  last = pos;

  pos = NULL;
  css_for_each_child(tmp, last)
   pos = tmp;
 } while (pos);

 return last;
}

static struct cgroup_subsys_state *
css_leftmost_descendant(struct cgroup_subsys_state *pos)
{
 struct cgroup_subsys_state *last;

 do {
  last = pos;
  pos = css_next_child(NULL, pos);
 } while (pos);

 return last;
}
struct cgroup_subsys_state *
css_next_descendant_post(struct cgroup_subsys_state *pos,
    struct cgroup_subsys_state *root)
{
 struct cgroup_subsys_state *next;

 cgroup_assert_mutex_or_rcu_locked();


 if (!pos)
  return css_leftmost_descendant(root);


 if (pos == root)
  return NULL;


 next = css_next_child(pos, pos->parent);
 if (next)
  return css_leftmost_descendant(next);


 return pos->parent;
}
bool css_has_online_children(struct cgroup_subsys_state *css)
{
 struct cgroup_subsys_state *child;
 bool ret = false;

 rcu_read_lock();
 css_for_each_child(child, css) {
  if (child->flags & CSS_ONLINE) {
   ret = true;
   break;
  }
 }
 rcu_read_unlock();
 return ret;
}







static void css_task_iter_advance_css_set(struct css_task_iter *it)
{
 struct list_head *l = it->cset_pos;
 struct cgrp_cset_link *link;
 struct css_set *cset;

 lockdep_assert_held(&css_set_lock);


 do {
  l = l->next;
  if (l == it->cset_head) {
   it->cset_pos = NULL;
   it->task_pos = NULL;
   return;
  }

  if (it->ss) {
   cset = container_of(l, struct css_set,
         e_cset_node[it->ss->id]);
  } else {
   link = list_entry(l, struct cgrp_cset_link, cset_link);
   cset = link->cset;
  }
 } while (!css_set_populated(cset));

 it->cset_pos = l;

 if (!list_empty(&cset->tasks))
  it->task_pos = cset->tasks.next;
 else
  it->task_pos = cset->mg_tasks.next;

 it->tasks_head = &cset->tasks;
 it->mg_tasks_head = &cset->mg_tasks;
 if (it->cur_cset) {
  list_del(&it->iters_node);
  put_css_set_locked(it->cur_cset);
 }
 get_css_set(cset);
 it->cur_cset = cset;
 list_add(&it->iters_node, &cset->task_iters);
}

static void css_task_iter_advance(struct css_task_iter *it)
{
 struct list_head *l = it->task_pos;

 lockdep_assert_held(&css_set_lock);
 WARN_ON_ONCE(!l);






 l = l->next;

 if (l == it->tasks_head)
  l = it->mg_tasks_head->next;

 if (l == it->mg_tasks_head)
  css_task_iter_advance_css_set(it);
 else
  it->task_pos = l;
}
void css_task_iter_start(struct cgroup_subsys_state *css,
    struct css_task_iter *it)
{

 WARN_ON_ONCE(!use_task_css_set_links);

 memset(it, 0, sizeof(*it));

 spin_lock_irq(&css_set_lock);

 it->ss = css->ss;

 if (it->ss)
  it->cset_pos = &css->cgroup->e_csets[css->ss->id];
 else
  it->cset_pos = &css->cgroup->cset_links;

 it->cset_head = it->cset_pos;

 css_task_iter_advance_css_set(it);

 spin_unlock_irq(&css_set_lock);
}
struct task_struct *css_task_iter_next(struct css_task_iter *it)
{
 if (it->cur_task) {
  put_task_struct(it->cur_task);
  it->cur_task = NULL;
 }

 spin_lock_irq(&css_set_lock);

 if (it->task_pos) {
  it->cur_task = list_entry(it->task_pos, struct task_struct,
       cg_list);
  get_task_struct(it->cur_task);
  css_task_iter_advance(it);
 }

 spin_unlock_irq(&css_set_lock);

 return it->cur_task;
}







void css_task_iter_end(struct css_task_iter *it)
{
 if (it->cur_cset) {
  spin_lock_irq(&css_set_lock);
  list_del(&it->iters_node);
  put_css_set_locked(it->cur_cset);
  spin_unlock_irq(&css_set_lock);
 }

 if (it->cur_task)
  put_task_struct(it->cur_task);
}
int cgroup_transfer_tasks(struct cgroup *to, struct cgroup *from)
{
 LIST_HEAD(preloaded_csets);
 struct cgrp_cset_link *link;
 struct css_task_iter it;
 struct task_struct *task;
 int ret;

 if (!cgroup_may_migrate_to(to))
  return -EBUSY;

 mutex_lock(&cgroup_mutex);


 spin_lock_irq(&css_set_lock);
 list_for_each_entry(link, &from->cset_links, cset_link)
  cgroup_migrate_add_src(link->cset, to, &preloaded_csets);
 spin_unlock_irq(&css_set_lock);

 ret = cgroup_migrate_prepare_dst(&preloaded_csets);
 if (ret)
  goto out_err;





 do {
  css_task_iter_start(&from->self, &it);
  task = css_task_iter_next(&it);
  if (task)
   get_task_struct(task);
  css_task_iter_end(&it);

  if (task) {
   ret = cgroup_migrate(task, false, to->root);
   put_task_struct(task);
  }
 } while (task && !ret);
out_err:
 cgroup_migrate_finish(&preloaded_csets);
 mutex_unlock(&cgroup_mutex);
 return ret;
}
enum cgroup_filetype {
 CGROUP_FILE_PROCS,
 CGROUP_FILE_TASKS,
};







struct cgroup_pidlist {




 struct { enum cgroup_filetype type; struct pid_namespace *ns; } key;

 pid_t *list;

 int length;

 struct list_head links;

 struct cgroup *owner;

 struct delayed_work destroy_dwork;
};






static void *pidlist_allocate(int count)
{
 if (PIDLIST_TOO_LARGE(count))
  return vmalloc(count * sizeof(pid_t));
 else
  return kmalloc(count * sizeof(pid_t), GFP_KERNEL);
}

static void pidlist_free(void *p)
{
 kvfree(p);
}





static void cgroup_pidlist_destroy_all(struct cgroup *cgrp)
{
 struct cgroup_pidlist *l, *tmp_l;

 mutex_lock(&cgrp->pidlist_mutex);
 list_for_each_entry_safe(l, tmp_l, &cgrp->pidlists, links)
  mod_delayed_work(cgroup_pidlist_destroy_wq, &l->destroy_dwork, 0);
 mutex_unlock(&cgrp->pidlist_mutex);

 flush_workqueue(cgroup_pidlist_destroy_wq);
 BUG_ON(!list_empty(&cgrp->pidlists));
}

static void cgroup_pidlist_destroy_work_fn(struct work_struct *work)
{
 struct delayed_work *dwork = to_delayed_work(work);
 struct cgroup_pidlist *l = container_of(dwork, struct cgroup_pidlist,
      destroy_dwork);
 struct cgroup_pidlist *tofree = NULL;

 mutex_lock(&l->owner->pidlist_mutex);





 if (!delayed_work_pending(dwork)) {
  list_del(&l->links);
  pidlist_free(l->list);
  put_pid_ns(l->key.ns);
  tofree = l;
 }

 mutex_unlock(&l->owner->pidlist_mutex);
 kfree(tofree);
}





static int pidlist_uniq(pid_t *list, int length)
{
 int src, dest = 1;





 if (length == 0 || length == 1)
  return length;

 for (src = 1; src < length; src++) {

  while (list[src] == list[src-1]) {
   src++;
   if (src == length)
    goto after;
  }

  list[dest] = list[src];
  dest++;
 }
after:
 return dest;
}
static pid_t pid_fry(pid_t pid)
{
 unsigned a = pid & 0x55555555;
 unsigned b = pid & 0xAAAAAAAA;

 return (a << 1) | (b >> 1);
}

static pid_t cgroup_pid_fry(struct cgroup *cgrp, pid_t pid)
{
 if (cgroup_on_dfl(cgrp))
  return pid_fry(pid);
 else
  return pid;
}

static int cmppid(const void *a, const void *b)
{
 return *(pid_t *)a - *(pid_t *)b;
}

static int fried_cmppid(const void *a, const void *b)
{
 return pid_fry(*(pid_t *)a) - pid_fry(*(pid_t *)b);
}

static struct cgroup_pidlist *cgroup_pidlist_find(struct cgroup *cgrp,
        enum cgroup_filetype type)
{
 struct cgroup_pidlist *l;

 struct pid_namespace *ns = task_active_pid_ns(current);

 lockdep_assert_held(&cgrp->pidlist_mutex);

 list_for_each_entry(l, &cgrp->pidlists, links)
  if (l->key.type == type && l->key.ns == ns)
   return l;
 return NULL;
}







static struct cgroup_pidlist *cgroup_pidlist_find_create(struct cgroup *cgrp,
      enum cgroup_filetype type)
{
 struct cgroup_pidlist *l;

 lockdep_assert_held(&cgrp->pidlist_mutex);

 l = cgroup_pidlist_find(cgrp, type);
 if (l)
  return l;


 l = kzalloc(sizeof(struct cgroup_pidlist), GFP_KERNEL);
 if (!l)
  return l;

 INIT_DELAYED_WORK(&l->destroy_dwork, cgroup_pidlist_destroy_work_fn);
 l->key.type = type;

 l->key.ns = get_pid_ns(task_active_pid_ns(current));
 l->owner = cgrp;
 list_add(&l->links, &cgrp->pidlists);
 return l;
}




static int pidlist_array_load(struct cgroup *cgrp, enum cgroup_filetype type,
         struct cgroup_pidlist **lp)
{
 pid_t *array;
 int length;
 int pid, n = 0;
 struct css_task_iter it;
 struct task_struct *tsk;
 struct cgroup_pidlist *l;

 lockdep_assert_held(&cgrp->pidlist_mutex);







 length = cgroup_task_count(cgrp);
 array = pidlist_allocate(length);
 if (!array)
  return -ENOMEM;

 css_task_iter_start(&cgrp->self, &it);
 while ((tsk = css_task_iter_next(&it))) {
  if (unlikely(n == length))
   break;

  if (type == CGROUP_FILE_PROCS)
   pid = task_tgid_vnr(tsk);
  else
   pid = task_pid_vnr(tsk);
  if (pid > 0)
   array[n++] = pid;
 }
 css_task_iter_end(&it);
 length = n;

 if (cgroup_on_dfl(cgrp))
  sort(array, length, sizeof(pid_t), fried_cmppid, NULL);
 else
  sort(array, length, sizeof(pid_t), cmppid, NULL);
 if (type == CGROUP_FILE_PROCS)
  length = pidlist_uniq(array, length);

 l = cgroup_pidlist_find_create(cgrp, type);
 if (!l) {
  pidlist_free(array);
  return -ENOMEM;
 }


 pidlist_free(l->list);
 l->list = array;
 l->length = length;
 *lp = l;
 return 0;
}
int cgroupstats_build(struct cgroupstats *stats, struct dentry *dentry)
{
 struct kernfs_node *kn = kernfs_node_from_dentry(dentry);
 struct cgroup *cgrp;
 struct css_task_iter it;
 struct task_struct *tsk;


 if (dentry->d_sb->s_type != &cgroup_fs_type || !kn ||
     kernfs_type(kn) != KERNFS_DIR)
  return -EINVAL;

 mutex_lock(&cgroup_mutex);






 rcu_read_lock();
 cgrp = rcu_dereference(kn->priv);
 if (!cgrp || cgroup_is_dead(cgrp)) {
  rcu_read_unlock();
  mutex_unlock(&cgroup_mutex);
  return -ENOENT;
 }
 rcu_read_unlock();

 css_task_iter_start(&cgrp->self, &it);
 while ((tsk = css_task_iter_next(&it))) {
  switch (tsk->state) {
  case TASK_RUNNING:
   stats->nr_running++;
   break;
  case TASK_INTERRUPTIBLE:
   stats->nr_sleeping++;
   break;
  case TASK_UNINTERRUPTIBLE:
   stats->nr_uninterruptible++;
   break;
  case TASK_STOPPED:
   stats->nr_stopped++;
   break;
  default:
   if (delayacct_is_task_waiting_on_io(tsk))
    stats->nr_io_wait++;
   break;
  }
 }
 css_task_iter_end(&it);

 mutex_unlock(&cgroup_mutex);
 return 0;
}
static void *cgroup_pidlist_start(struct seq_file *s, loff_t *pos)
{






 struct kernfs_open_file *of = s->private;
 struct cgroup *cgrp = seq_css(s)->cgroup;
 struct cgroup_pidlist *l;
 enum cgroup_filetype type = seq_cft(s)->private;
 int index = 0, pid = *pos;
 int *iter, ret;

 mutex_lock(&cgrp->pidlist_mutex);







 if (of->priv)
  of->priv = cgroup_pidlist_find(cgrp, type);





 if (!of->priv) {
  ret = pidlist_array_load(cgrp, type,
      (struct cgroup_pidlist **)&of->priv);
  if (ret)
   return ERR_PTR(ret);
 }
 l = of->priv;

 if (pid) {
  int end = l->length;

  while (index < end) {
   int mid = (index + end) / 2;
   if (cgroup_pid_fry(cgrp, l->list[mid]) == pid) {
    index = mid;
    break;
   } else if (cgroup_pid_fry(cgrp, l->list[mid]) <= pid)
    index = mid + 1;
   else
    end = mid;
  }
 }

 if (index >= l->length)
  return NULL;

 iter = l->list + index;
 *pos = cgroup_pid_fry(cgrp, *iter);
 return iter;
}

static void cgroup_pidlist_stop(struct seq_file *s, void *v)
{
 struct kernfs_open_file *of = s->private;
 struct cgroup_pidlist *l = of->priv;

 if (l)
  mod_delayed_work(cgroup_pidlist_destroy_wq, &l->destroy_dwork,
     CGROUP_PIDLIST_DESTROY_DELAY);
 mutex_unlock(&seq_css(s)->cgroup->pidlist_mutex);
}

static void *cgroup_pidlist_next(struct seq_file *s, void *v, loff_t *pos)
{
 struct kernfs_open_file *of = s->private;
 struct cgroup_pidlist *l = of->priv;
 pid_t *p = v;
 pid_t *end = l->list + l->length;




 p++;
 if (p >= end) {
  return NULL;
 } else {
  *pos = cgroup_pid_fry(seq_css(s)->cgroup, *p);
  return p;
 }
}

static int cgroup_pidlist_show(struct seq_file *s, void *v)
{
 seq_printf(s, "%d\n", *(int *)v);

 return 0;
}

static u64 cgroup_read_notify_on_release(struct cgroup_subsys_state *css,
      struct cftype *cft)
{
 return notify_on_release(css->cgroup);
}

static int cgroup_write_notify_on_release(struct cgroup_subsys_state *css,
       struct cftype *cft, u64 val)
{
 if (val)
  set_bit(CGRP_NOTIFY_ON_RELEASE, &css->cgroup->flags);
 else
  clear_bit(CGRP_NOTIFY_ON_RELEASE, &css->cgroup->flags);
 return 0;
}

static u64 cgroup_clone_children_read(struct cgroup_subsys_state *css,
          struct cftype *cft)
{
 return test_bit(CGRP_CPUSET_CLONE_CHILDREN, &css->cgroup->flags);
}

static int cgroup_clone_children_write(struct cgroup_subsys_state *css,
           struct cftype *cft, u64 val)
{
 if (val)
  set_bit(CGRP_CPUSET_CLONE_CHILDREN, &css->cgroup->flags);
 else
  clear_bit(CGRP_CPUSET_CLONE_CHILDREN, &css->cgroup->flags);
 return 0;
}


static struct cftype cgroup_dfl_base_files[] = {
 {
  .name = "cgroup.procs",
  .file_offset = offsetof(struct cgroup, procs_file),
  .seq_start = cgroup_pidlist_start,
  .seq_next = cgroup_pidlist_next,
  .seq_stop = cgroup_pidlist_stop,
  .seq_show = cgroup_pidlist_show,
  .private = CGROUP_FILE_PROCS,
  .write = cgroup_procs_write,
 },
 {
  .name = "cgroup.controllers",
  .seq_show = cgroup_controllers_show,
 },
 {
  .name = "cgroup.subtree_control",
  .seq_show = cgroup_subtree_control_show,
  .write = cgroup_subtree_control_write,
 },
 {
  .name = "cgroup.events",
  .flags = CFTYPE_NOT_ON_ROOT,
  .file_offset = offsetof(struct cgroup, events_file),
  .seq_show = cgroup_events_show,
 },
 { }
};


static struct cftype cgroup_legacy_base_files[] = {
 {
  .name = "cgroup.procs",
  .seq_start = cgroup_pidlist_start,
  .seq_next = cgroup_pidlist_next,
  .seq_stop = cgroup_pidlist_stop,
  .seq_show = cgroup_pidlist_show,
  .private = CGROUP_FILE_PROCS,
  .write = cgroup_procs_write,
 },
 {
  .name = "cgroup.clone_children",
  .read_u64 = cgroup_clone_children_read,
  .write_u64 = cgroup_clone_children_write,
 },
 {
  .name = "cgroup.sane_behavior",
  .flags = CFTYPE_ONLY_ON_ROOT,
  .seq_show = cgroup_sane_behavior_show,
 },
 {
  .name = "tasks",
  .seq_start = cgroup_pidlist_start,
  .seq_next = cgroup_pidlist_next,
  .seq_stop = cgroup_pidlist_stop,
  .seq_show = cgroup_pidlist_show,
  .private = CGROUP_FILE_TASKS,
  .write = cgroup_tasks_write,
 },
 {
  .name = "notify_on_release",
  .read_u64 = cgroup_read_notify_on_release,
  .write_u64 = cgroup_write_notify_on_release,
 },
 {
  .name = "release_agent",
  .flags = CFTYPE_ONLY_ON_ROOT,
  .seq_show = cgroup_release_agent_show,
  .write = cgroup_release_agent_write,
  .max_write_len = PATH_MAX - 1,
 },
 { }
};
static void css_free_work_fn(struct work_struct *work)
{
 struct cgroup_subsys_state *css =
  container_of(work, struct cgroup_subsys_state, destroy_work);
 struct cgroup_subsys *ss = css->ss;
 struct cgroup *cgrp = css->cgroup;

 percpu_ref_exit(&css->refcnt);

 if (ss) {

  struct cgroup_subsys_state *parent = css->parent;
  int id = css->id;

  ss->css_free(css);
  cgroup_idr_remove(&ss->css_idr, id);
  cgroup_put(cgrp);

  if (parent)
   css_put(parent);
 } else {

  atomic_dec(&cgrp->root->nr_cgrps);
  cgroup_pidlist_destroy_all(cgrp);
  cancel_work_sync(&cgrp->release_agent_work);

  if (cgroup_parent(cgrp)) {






   cgroup_put(cgroup_parent(cgrp));
   kernfs_put(cgrp->kn);
   kfree(cgrp);
  } else {





   cgroup_destroy_root(cgrp->root);
  }
 }
}

static void css_free_rcu_fn(struct rcu_head *rcu_head)
{
 struct cgroup_subsys_state *css =
  container_of(rcu_head, struct cgroup_subsys_state, rcu_head);

 INIT_WORK(&css->destroy_work, css_free_work_fn);
 queue_work(cgroup_destroy_wq, &css->destroy_work);
}

static void css_release_work_fn(struct work_struct *work)
{
 struct cgroup_subsys_state *css =
  container_of(work, struct cgroup_subsys_state, destroy_work);
 struct cgroup_subsys *ss = css->ss;
 struct cgroup *cgrp = css->cgroup;

 mutex_lock(&cgroup_mutex);

 css->flags |= CSS_RELEASED;
 list_del_rcu(&css->sibling);

 if (ss) {

  cgroup_idr_replace(&ss->css_idr, NULL, css->id);
  if (ss->css_released)
   ss->css_released(css);
 } else {

  cgroup_idr_remove(&cgrp->root->cgroup_idr, cgrp->id);
  cgrp->id = -1;
  if (cgrp->kn)
   RCU_INIT_POINTER(*(void __rcu __force **)&cgrp->kn->priv,
      NULL);
 }

 mutex_unlock(&cgroup_mutex);

 call_rcu(&css->rcu_head, css_free_rcu_fn);
}

static void css_release(struct percpu_ref *ref)
{
 struct cgroup_subsys_state *css =
  container_of(ref, struct cgroup_subsys_state, refcnt);

 INIT_WORK(&css->destroy_work, css_release_work_fn);
 queue_work(cgroup_destroy_wq, &css->destroy_work);
}

static void init_and_link_css(struct cgroup_subsys_state *css,
         struct cgroup_subsys *ss, struct cgroup *cgrp)
{
 lockdep_assert_held(&cgroup_mutex);

 cgroup_get(cgrp);

 memset(css, 0, sizeof(*css));
 css->cgroup = cgrp;
 css->ss = ss;
 css->id = -1;
 INIT_LIST_HEAD(&css->sibling);
 INIT_LIST_HEAD(&css->children);
 css->serial_nr = css_serial_nr_next++;
 atomic_set(&css->online_cnt, 0);

 if (cgroup_parent(cgrp)) {
  css->parent = cgroup_css(cgroup_parent(cgrp), ss);
  css_get(css->parent);
 }

 BUG_ON(cgroup_css(cgrp, ss));
}


static int online_css(struct cgroup_subsys_state *css)
{
 struct cgroup_subsys *ss = css->ss;
 int ret = 0;

 lockdep_assert_held(&cgroup_mutex);

 if (ss->css_online)
  ret = ss->css_online(css);
 if (!ret) {
  css->flags |= CSS_ONLINE;
  rcu_assign_pointer(css->cgroup->subsys[ss->id], css);

  atomic_inc(&css->online_cnt);
  if (css->parent)
   atomic_inc(&css->parent->online_cnt);
 }
 return ret;
}


static void offline_css(struct cgroup_subsys_state *css)
{
 struct cgroup_subsys *ss = css->ss;

 lockdep_assert_held(&cgroup_mutex);

 if (!(css->flags & CSS_ONLINE))
  return;

 if (ss->css_reset)
  ss->css_reset(css);

 if (ss->css_offline)
  ss->css_offline(css);

 css->flags &= ~CSS_ONLINE;
 RCU_INIT_POINTER(css->cgroup->subsys[ss->id], NULL);

 wake_up_all(&css->cgroup->offline_waitq);
}
static struct cgroup_subsys_state *css_create(struct cgroup *cgrp,
           struct cgroup_subsys *ss)
{
 struct cgroup *parent = cgroup_parent(cgrp);
 struct cgroup_subsys_state *parent_css = cgroup_css(parent, ss);
 struct cgroup_subsys_state *css;
 int err;

 lockdep_assert_held(&cgroup_mutex);

 css = ss->css_alloc(parent_css);
 if (IS_ERR(css))
  return css;

 init_and_link_css(css, ss, cgrp);

 err = percpu_ref_init(&css->refcnt, css_release, 0, GFP_KERNEL);
 if (err)
  goto err_free_css;

 err = cgroup_idr_alloc(&ss->css_idr, NULL, 2, 0, GFP_KERNEL);
 if (err < 0)
  goto err_free_css;
 css->id = err;


 list_add_tail_rcu(&css->sibling, &parent_css->children);
 cgroup_idr_replace(&ss->css_idr, css, css->id);

 err = online_css(css);
 if (err)
  goto err_list_del;

 if (ss->broken_hierarchy && !ss->warned_broken_hierarchy &&
     cgroup_parent(parent)) {
  pr_warn("%s (%d) created nested cgroup for controller \"%s\" which has incomplete hierarchy support. Nested cgroups may change behavior in the future.\n",
   current->comm, current->pid, ss->name);
  if (!strcmp(ss->name, "memory"))
   pr_warn("\"memory\" requires setting use_hierarchy to 1 on the root\n");
  ss->warned_broken_hierarchy = true;
 }

 return css;

err_list_del:
 list_del_rcu(&css->sibling);
err_free_css:
 call_rcu(&css->rcu_head, css_free_rcu_fn);
 return ERR_PTR(err);
}

static struct cgroup *cgroup_create(struct cgroup *parent)
{
 struct cgroup_root *root = parent->root;
 struct cgroup *cgrp, *tcgrp;
 int level = parent->level + 1;
 int ret;


 cgrp = kzalloc(sizeof(*cgrp) +
         sizeof(cgrp->ancestor_ids[0]) * (level + 1), GFP_KERNEL);
 if (!cgrp)
  return ERR_PTR(-ENOMEM);

 ret = percpu_ref_init(&cgrp->self.refcnt, css_release, 0, GFP_KERNEL);
 if (ret)
  goto out_free_cgrp;





 cgrp->id = cgroup_idr_alloc(&root->cgroup_idr, NULL, 2, 0, GFP_KERNEL);
 if (cgrp->id < 0) {
  ret = -ENOMEM;
  goto out_cancel_ref;
 }

 init_cgroup_housekeeping(cgrp);

 cgrp->self.parent = &parent->self;
 cgrp->root = root;
 cgrp->level = level;

 for (tcgrp = cgrp; tcgrp; tcgrp = cgroup_parent(tcgrp))
  cgrp->ancestor_ids[tcgrp->level] = tcgrp->id;

 if (notify_on_release(parent))
  set_bit(CGRP_NOTIFY_ON_RELEASE, &cgrp->flags);

 if (test_bit(CGRP_CPUSET_CLONE_CHILDREN, &parent->flags))
  set_bit(CGRP_CPUSET_CLONE_CHILDREN, &cgrp->flags);

 cgrp->self.serial_nr = css_serial_nr_next++;


 list_add_tail_rcu(&cgrp->self.sibling, &cgroup_parent(cgrp)->self.children);
 atomic_inc(&root->nr_cgrps);
 cgroup_get(parent);





 cgroup_idr_replace(&root->cgroup_idr, cgrp, cgrp->id);





 if (!cgroup_on_dfl(cgrp))
  cgrp->subtree_control = cgroup_control(cgrp);

 cgroup_propagate_control(cgrp);


 ret = cgroup_apply_control_enable(cgrp);
 if (ret)
  goto out_destroy;

 return cgrp;

out_cancel_ref:
 percpu_ref_exit(&cgrp->self.refcnt);
out_free_cgrp:
 kfree(cgrp);
 return ERR_PTR(ret);
out_destroy:
 cgroup_destroy_locked(cgrp);
 return ERR_PTR(ret);
}

static int cgroup_mkdir(struct kernfs_node *parent_kn, const char *name,
   umode_t mode)
{
 struct cgroup *parent, *cgrp;
 struct kernfs_node *kn;
 int ret;


 if (strchr(name, '\n'))
  return -EINVAL;

 parent = cgroup_kn_lock_live(parent_kn, false);
 if (!parent)
  return -ENODEV;

 cgrp = cgroup_create(parent);
 if (IS_ERR(cgrp)) {
  ret = PTR_ERR(cgrp);
  goto out_unlock;
 }


 kn = kernfs_create_dir(parent->kn, name, mode, cgrp);
 if (IS_ERR(kn)) {
  ret = PTR_ERR(kn);
  goto out_destroy;
 }
 cgrp->kn = kn;





 kernfs_get(kn);

 ret = cgroup_kn_set_ugid(kn);
 if (ret)
  goto out_destroy;

 ret = css_populate_dir(&cgrp->self);
 if (ret)
  goto out_destroy;

 ret = cgroup_apply_control_enable(cgrp);
 if (ret)
  goto out_destroy;


 kernfs_activate(kn);

 ret = 0;
 goto out_unlock;

out_destroy:
 cgroup_destroy_locked(cgrp);
out_unlock:
 cgroup_kn_unlock(parent_kn);
 return ret;
}






static void css_killed_work_fn(struct work_struct *work)
{
 struct cgroup_subsys_state *css =
  container_of(work, struct cgroup_subsys_state, destroy_work);

 mutex_lock(&cgroup_mutex);

 do {
  offline_css(css);
  css_put(css);

  css = css->parent;
 } while (css && atomic_dec_and_test(&css->online_cnt));

 mutex_unlock(&cgroup_mutex);
}


static void css_killed_ref_fn(struct percpu_ref *ref)
{
 struct cgroup_subsys_state *css =
  container_of(ref, struct cgroup_subsys_state, refcnt);

 if (atomic_dec_and_test(&css->online_cnt)) {
  INIT_WORK(&css->destroy_work, css_killed_work_fn);
  queue_work(cgroup_destroy_wq, &css->destroy_work);
 }
}
static void kill_css(struct cgroup_subsys_state *css)
{
 lockdep_assert_held(&cgroup_mutex);





 css_clear_dir(css);





 css_get(css);
 percpu_ref_kill_and_confirm(&css->refcnt, css_killed_ref_fn);
}
static int cgroup_destroy_locked(struct cgroup *cgrp)
 __releases(&cgroup_mutex) __acquires(&cgroup_mutex)
{
 struct cgroup_subsys_state *css;
 struct cgrp_cset_link *link;
 int ssid;

 lockdep_assert_held(&cgroup_mutex);





 if (cgroup_is_populated(cgrp))
  return -EBUSY;






 if (css_has_online_children(&cgrp->self))
  return -EBUSY;







 cgrp->self.flags &= ~CSS_ONLINE;

 spin_lock_irq(&css_set_lock);
 list_for_each_entry(link, &cgrp->cset_links, cset_link)
  link->cset->dead = true;
 spin_unlock_irq(&css_set_lock);


 for_each_css(css, ssid, cgrp)
  kill_css(css);





 kernfs_remove(cgrp->kn);

 check_for_release(cgroup_parent(cgrp));


 percpu_ref_kill(&cgrp->self.refcnt);

 return 0;
};

static int cgroup_rmdir(struct kernfs_node *kn)
{
 struct cgroup *cgrp;
 int ret = 0;

 cgrp = cgroup_kn_lock_live(kn, false);
 if (!cgrp)
  return 0;

 ret = cgroup_destroy_locked(cgrp);

 cgroup_kn_unlock(kn);
 return ret;
}

static struct kernfs_syscall_ops cgroup_kf_syscall_ops = {
 .remount_fs = cgroup_remount,
 .show_options = cgroup_show_options,
 .mkdir = cgroup_mkdir,
 .rmdir = cgroup_rmdir,
 .rename = cgroup_rename,
 .show_path = cgroup_show_path,
};

static void __init cgroup_init_subsys(struct cgroup_subsys *ss, bool early)
{
 struct cgroup_subsys_state *css;

 pr_debug("Initializing cgroup subsys %s\n", ss->name);

 mutex_lock(&cgroup_mutex);

 idr_init(&ss->css_idr);
 INIT_LIST_HEAD(&ss->cfts);


 ss->root = &cgrp_dfl_root;
 css = ss->css_alloc(cgroup_css(&cgrp_dfl_root.cgrp, ss));

 BUG_ON(IS_ERR(css));
 init_and_link_css(css, ss, &cgrp_dfl_root.cgrp);





 css->flags |= CSS_NO_REF;

 if (early) {

  css->id = 1;
 } else {
  css->id = cgroup_idr_alloc(&ss->css_idr, css, 1, 2, GFP_KERNEL);
  BUG_ON(css->id < 0);
 }





 init_css_set.subsys[ss->id] = css;

 have_fork_callback |= (bool)ss->fork << ss->id;
 have_exit_callback |= (bool)ss->exit << ss->id;
 have_free_callback |= (bool)ss->free << ss->id;
 have_canfork_callback |= (bool)ss->can_fork << ss->id;




 BUG_ON(!list_empty(&init_task.tasks));

 BUG_ON(online_css(css));

 mutex_unlock(&cgroup_mutex);
}







int __init cgroup_init_early(void)
{
 static struct cgroup_sb_opts __initdata opts;
 struct cgroup_subsys *ss;
 int i;

 init_cgroup_root(&cgrp_dfl_root, &opts);
 cgrp_dfl_root.cgrp.self.flags |= CSS_NO_REF;

 RCU_INIT_POINTER(init_task.cgroups, &init_css_set);

 for_each_subsys(ss, i) {
  WARN(!ss->css_alloc || !ss->css_free || ss->name || ss->id,
       "invalid cgroup_subsys %d:%s css_alloc=%p css_free=%p id:name=%d:%s\n",
       i, cgroup_subsys_name[i], ss->css_alloc, ss->css_free,
       ss->id, ss->name);
  WARN(strlen(cgroup_subsys_name[i]) > MAX_CGROUP_TYPE_NAMELEN,
       "cgroup_subsys_name %s too long\n", cgroup_subsys_name[i]);

  ss->id = i;
  ss->name = cgroup_subsys_name[i];
  if (!ss->legacy_name)
   ss->legacy_name = cgroup_subsys_name[i];

  if (ss->early_init)
   cgroup_init_subsys(ss, true);
 }
 return 0;
}

static u16 cgroup_disable_mask __initdata;







int __init cgroup_init(void)
{
 struct cgroup_subsys *ss;
 int ssid;

 BUILD_BUG_ON(CGROUP_SUBSYS_COUNT > 16);
 BUG_ON(percpu_init_rwsem(&cgroup_threadgroup_rwsem));
 BUG_ON(cgroup_init_cftypes(NULL, cgroup_dfl_base_files));
 BUG_ON(cgroup_init_cftypes(NULL, cgroup_legacy_base_files));

 get_user_ns(init_cgroup_ns.user_ns);

 mutex_lock(&cgroup_mutex);





 hash_add(css_set_table, &init_css_set.hlist,
   css_set_hash(init_css_set.subsys));

 BUG_ON(cgroup_setup_root(&cgrp_dfl_root, 0));

 mutex_unlock(&cgroup_mutex);

 for_each_subsys(ss, ssid) {
  if (ss->early_init) {
   struct cgroup_subsys_state *css =
    init_css_set.subsys[ss->id];

   css->id = cgroup_idr_alloc(&ss->css_idr, css, 1, 2,
         GFP_KERNEL);
   BUG_ON(css->id < 0);
  } else {
   cgroup_init_subsys(ss, false);
  }

  list_add_tail(&init_css_set.e_cset_node[ssid],
         &cgrp_dfl_root.cgrp.e_csets[ssid]);






  if (cgroup_disable_mask & (1 << ssid)) {
   static_branch_disable(cgroup_subsys_enabled_key[ssid]);
   printk(KERN_INFO "Disabling %s control group subsystem\n",
          ss->name);
   continue;
  }

  if (cgroup_ssid_no_v1(ssid))
   printk(KERN_INFO "Disabling %s control group subsystem in v1 mounts\n",
          ss->name);

  cgrp_dfl_root.subsys_mask |= 1 << ss->id;

  if (ss->implicit_on_dfl)
   cgrp_dfl_implicit_ss_mask |= 1 << ss->id;
  else if (!ss->dfl_cftypes)
   cgrp_dfl_inhibit_ss_mask |= 1 << ss->id;

  if (ss->dfl_cftypes == ss->legacy_cftypes) {
   WARN_ON(cgroup_add_cftypes(ss, ss->dfl_cftypes));
  } else {
   WARN_ON(cgroup_add_dfl_cftypes(ss, ss->dfl_cftypes));
   WARN_ON(cgroup_add_legacy_cftypes(ss, ss->legacy_cftypes));
  }

  if (ss->bind)
   ss->bind(init_css_set.subsys[ssid]);
 }


 hash_del(&init_css_set.hlist);
 hash_add(css_set_table, &init_css_set.hlist,
   css_set_hash(init_css_set.subsys));

 WARN_ON(sysfs_create_mount_point(fs_kobj, "cgroup"));
 WARN_ON(register_filesystem(&cgroup_fs_type));
 WARN_ON(register_filesystem(&cgroup2_fs_type));
 WARN_ON(!proc_create("cgroups", 0, NULL, &proc_cgroupstats_operations));

 return 0;
}

static int __init cgroup_wq_init(void)
{
 cgroup_destroy_wq = alloc_workqueue("cgroup_destroy", 0, 1);
 BUG_ON(!cgroup_destroy_wq);





 cgroup_pidlist_destroy_wq = alloc_workqueue("cgroup_pidlist_destroy",
          0, 1);
 BUG_ON(!cgroup_pidlist_destroy_wq);

 return 0;
}
core_initcall(cgroup_wq_init);






int proc_cgroup_show(struct seq_file *m, struct pid_namespace *ns,
       struct pid *pid, struct task_struct *tsk)
{
 char *buf, *path;
 int retval;
 struct cgroup_root *root;

 retval = -ENOMEM;
 buf = kmalloc(PATH_MAX, GFP_KERNEL);
 if (!buf)
  goto out;

 mutex_lock(&cgroup_mutex);
 spin_lock_irq(&css_set_lock);

 for_each_root(root) {
  struct cgroup_subsys *ss;
  struct cgroup *cgrp;
  int ssid, count = 0;

  if (root == &cgrp_dfl_root && !cgrp_dfl_visible)
   continue;

  seq_printf(m, "%d:", root->hierarchy_id);
  if (root != &cgrp_dfl_root)
   for_each_subsys(ss, ssid)
    if (root->subsys_mask & (1 << ssid))
     seq_printf(m, "%s%s", count++ ? "," : "",
         ss->legacy_name);
  if (strlen(root->name))
   seq_printf(m, "%sname=%s", count ? "," : "",
       root->name);
  seq_putc(m, ':');

  cgrp = task_cgroup_from_root(tsk, root);
  if (cgroup_on_dfl(cgrp) || !(tsk->flags & PF_EXITING)) {
   path = cgroup_path_ns_locked(cgrp, buf, PATH_MAX,
      current->nsproxy->cgroup_ns);
   if (!path) {
    retval = -ENAMETOOLONG;
    goto out_unlock;
   }
  } else {
   path = "/";
  }

  seq_puts(m, path);

  if (cgroup_on_dfl(cgrp) && cgroup_is_dead(cgrp))
   seq_puts(m, " (deleted)\n");
  else
   seq_putc(m, '\n');
 }

 retval = 0;
out_unlock:
 spin_unlock_irq(&css_set_lock);
 mutex_unlock(&cgroup_mutex);
 kfree(buf);
out:
 return retval;
}


static int proc_cgroupstats_show(struct seq_file *m, void *v)
{
 struct cgroup_subsys *ss;
 int i;

 seq_puts(m, "#subsys_name\thierarchy\tnum_cgroups\tenabled\n");





 mutex_lock(&cgroup_mutex);

 for_each_subsys(ss, i)
  seq_printf(m, "%s\t%d\t%d\t%d\n",
      ss->legacy_name, ss->root->hierarchy_id,
      atomic_read(&ss->root->nr_cgrps),
      cgroup_ssid_enabled(i));

 mutex_unlock(&cgroup_mutex);
 return 0;
}

static int cgroupstats_open(struct inode *inode, struct file *file)
{
 return single_open(file, proc_cgroupstats_show, NULL);
}

static const struct file_operations proc_cgroupstats_operations = {
 .open = cgroupstats_open,
 .read = seq_read,
 .llseek = seq_lseek,
 .release = single_release,
};
void cgroup_fork(struct task_struct *child)
{
 RCU_INIT_POINTER(child->cgroups, &init_css_set);
 INIT_LIST_HEAD(&child->cg_list);
}
int cgroup_can_fork(struct task_struct *child)
{
 struct cgroup_subsys *ss;
 int i, j, ret;

 do_each_subsys_mask(ss, i, have_canfork_callback) {
  ret = ss->can_fork(child);
  if (ret)
   goto out_revert;
 } while_each_subsys_mask();

 return 0;

out_revert:
 for_each_subsys(ss, j) {
  if (j >= i)
   break;
  if (ss->cancel_fork)
   ss->cancel_fork(child);
 }

 return ret;
}
void cgroup_cancel_fork(struct task_struct *child)
{
 struct cgroup_subsys *ss;
 int i;

 for_each_subsys(ss, i)
  if (ss->cancel_fork)
   ss->cancel_fork(child);
}
void cgroup_post_fork(struct task_struct *child)
{
 struct cgroup_subsys *ss;
 int i;
 if (use_task_css_set_links) {
  struct css_set *cset;

  spin_lock_irq(&css_set_lock);
  cset = task_css_set(current);
  if (list_empty(&child->cg_list)) {
   get_css_set(cset);
   css_set_move_task(child, NULL, cset, false);
  }
  spin_unlock_irq(&css_set_lock);
 }






 do_each_subsys_mask(ss, i, have_fork_callback) {
  ss->fork(child);
 } while_each_subsys_mask();
}
void cgroup_exit(struct task_struct *tsk)
{
 struct cgroup_subsys *ss;
 struct css_set *cset;
 int i;





 cset = task_css_set(tsk);

 if (!list_empty(&tsk->cg_list)) {
  spin_lock_irq(&css_set_lock);
  css_set_move_task(tsk, cset, NULL, false);
  spin_unlock_irq(&css_set_lock);
 } else {
  get_css_set(cset);
 }


 do_each_subsys_mask(ss, i, have_exit_callback) {
  ss->exit(tsk);
 } while_each_subsys_mask();
}

void cgroup_free(struct task_struct *task)
{
 struct css_set *cset = task_css_set(task);
 struct cgroup_subsys *ss;
 int ssid;

 do_each_subsys_mask(ss, ssid, have_free_callback) {
  ss->free(task);
 } while_each_subsys_mask();

 put_css_set(cset);
}

static void check_for_release(struct cgroup *cgrp)
{
 if (notify_on_release(cgrp) && !cgroup_is_populated(cgrp) &&
     !css_has_online_children(&cgrp->self) && !cgroup_is_dead(cgrp))
  schedule_work(&cgrp->release_agent_work);
}
static void cgroup_release_agent(struct work_struct *work)
{
 struct cgroup *cgrp =
  container_of(work, struct cgroup, release_agent_work);
 char *pathbuf = NULL, *agentbuf = NULL, *path;
 char *argv[3], *envp[3];

 mutex_lock(&cgroup_mutex);

 pathbuf = kmalloc(PATH_MAX, GFP_KERNEL);
 agentbuf = kstrdup(cgrp->root->release_agent_path, GFP_KERNEL);
 if (!pathbuf || !agentbuf)
  goto out;

 spin_lock_irq(&css_set_lock);
 path = cgroup_path_ns_locked(cgrp, pathbuf, PATH_MAX, &init_cgroup_ns);
 spin_unlock_irq(&css_set_lock);
 if (!path)
  goto out;

 argv[0] = agentbuf;
 argv[1] = path;
 argv[2] = NULL;


 envp[0] = "HOME=/";
 envp[1] = "PATH=/sbin:/bin:/usr/sbin:/usr/bin";
 envp[2] = NULL;

 mutex_unlock(&cgroup_mutex);
 call_usermodehelper(argv[0], argv, envp, UMH_WAIT_EXEC);
 goto out_free;
out:
 mutex_unlock(&cgroup_mutex);
out_free:
 kfree(agentbuf);
 kfree(pathbuf);
}

static int __init cgroup_disable(char *str)
{
 struct cgroup_subsys *ss;
 char *token;
 int i;

 while ((token = strsep(&str, ",")) != NULL) {
  if (!*token)
   continue;

  for_each_subsys(ss, i) {
   if (strcmp(token, ss->name) &&
       strcmp(token, ss->legacy_name))
    continue;
   cgroup_disable_mask |= 1 << i;
  }
 }
 return 1;
}
__setup("cgroup_disable=", cgroup_disable);

static int __init cgroup_no_v1(char *str)
{
 struct cgroup_subsys *ss;
 char *token;
 int i;

 while ((token = strsep(&str, ",")) != NULL) {
  if (!*token)
   continue;

  if (!strcmp(token, "all")) {
   cgroup_no_v1_mask = U16_MAX;
   break;
  }

  for_each_subsys(ss, i) {
   if (strcmp(token, ss->name) &&
       strcmp(token, ss->legacy_name))
    continue;

   cgroup_no_v1_mask |= 1 << i;
  }
 }
 return 1;
}
__setup("cgroup_no_v1=", cgroup_no_v1);
struct cgroup_subsys_state *css_tryget_online_from_dir(struct dentry *dentry,
             struct cgroup_subsys *ss)
{
 struct kernfs_node *kn = kernfs_node_from_dentry(dentry);
 struct file_system_type *s_type = dentry->d_sb->s_type;
 struct cgroup_subsys_state *css = NULL;
 struct cgroup *cgrp;


 if ((s_type != &cgroup_fs_type && s_type != &cgroup2_fs_type) ||
     !kn || kernfs_type(kn) != KERNFS_DIR)
  return ERR_PTR(-EBADF);

 rcu_read_lock();






 cgrp = rcu_dereference(kn->priv);
 if (cgrp)
  css = cgroup_css(cgrp, ss);

 if (!css || !css_tryget_online(css))
  css = ERR_PTR(-ENOENT);

 rcu_read_unlock();
 return css;
}
struct cgroup_subsys_state *css_from_id(int id, struct cgroup_subsys *ss)
{
 WARN_ON_ONCE(!rcu_read_lock_held());
 return id > 0 ? idr_find(&ss->css_idr, id) : NULL;
}
struct cgroup *cgroup_get_from_path(const char *path)
{
 struct kernfs_node *kn;
 struct cgroup *cgrp;

 mutex_lock(&cgroup_mutex);

 kn = kernfs_walk_and_get(cgrp_dfl_root.cgrp.kn, path);
 if (kn) {
  if (kernfs_type(kn) == KERNFS_DIR) {
   cgrp = kn->priv;
   cgroup_get(cgrp);
  } else {
   cgrp = ERR_PTR(-ENOTDIR);
  }
  kernfs_put(kn);
 } else {
  cgrp = ERR_PTR(-ENOENT);
 }

 mutex_unlock(&cgroup_mutex);
 return cgrp;
}
EXPORT_SYMBOL_GPL(cgroup_get_from_path);







DEFINE_SPINLOCK(cgroup_sk_update_lock);
static bool cgroup_sk_alloc_disabled __read_mostly;

void cgroup_sk_alloc_disable(void)
{
 if (cgroup_sk_alloc_disabled)
  return;
 pr_info("cgroup: disabling cgroup2 socket matching due to net_prio or net_cls activation\n");
 cgroup_sk_alloc_disabled = true;
}




void cgroup_sk_alloc(struct sock_cgroup_data *skcd)
{
 if (cgroup_sk_alloc_disabled)
  return;

 rcu_read_lock();

 while (true) {
  struct css_set *cset;

  cset = task_css_set(current);
  if (likely(cgroup_tryget(cset->dfl_cgrp))) {
   skcd->val = (unsigned long)cset->dfl_cgrp;
   break;
  }
  cpu_relax();
 }

 rcu_read_unlock();
}

void cgroup_sk_free(struct sock_cgroup_data *skcd)
{
 cgroup_put(sock_cgroup_ptr(skcd));
}




static struct cgroup_namespace *alloc_cgroup_ns(void)
{
 struct cgroup_namespace *new_ns;
 int ret;

 new_ns = kzalloc(sizeof(struct cgroup_namespace), GFP_KERNEL);
 if (!new_ns)
  return ERR_PTR(-ENOMEM);
 ret = ns_alloc_inum(&new_ns->ns);
 if (ret) {
  kfree(new_ns);
  return ERR_PTR(ret);
 }
 atomic_set(&new_ns->count, 1);
 new_ns->ns.ops = &cgroupns_operations;
 return new_ns;
}

void free_cgroup_ns(struct cgroup_namespace *ns)
{
 put_css_set(ns->root_cset);
 put_user_ns(ns->user_ns);
 ns_free_inum(&ns->ns);
 kfree(ns);
}
EXPORT_SYMBOL(free_cgroup_ns);

struct cgroup_namespace *copy_cgroup_ns(unsigned long flags,
     struct user_namespace *user_ns,
     struct cgroup_namespace *old_ns)
{
 struct cgroup_namespace *new_ns;
 struct css_set *cset;

 BUG_ON(!old_ns);

 if (!(flags & CLONE_NEWCGROUP)) {
  get_cgroup_ns(old_ns);
  return old_ns;
 }


 if (!ns_capable(user_ns, CAP_SYS_ADMIN))
  return ERR_PTR(-EPERM);

 mutex_lock(&cgroup_mutex);
 spin_lock_irq(&css_set_lock);

 cset = task_css_set(current);
 get_css_set(cset);

 spin_unlock_irq(&css_set_lock);
 mutex_unlock(&cgroup_mutex);

 new_ns = alloc_cgroup_ns();
 if (IS_ERR(new_ns)) {
  put_css_set(cset);
  return new_ns;
 }

 new_ns->user_ns = get_user_ns(user_ns);
 new_ns->root_cset = cset;

 return new_ns;
}

static inline struct cgroup_namespace *to_cg_ns(struct ns_common *ns)
{
 return container_of(ns, struct cgroup_namespace, ns);
}

static int cgroupns_install(struct nsproxy *nsproxy, struct ns_common *ns)
{
 struct cgroup_namespace *cgroup_ns = to_cg_ns(ns);

 if (!ns_capable(current_user_ns(), CAP_SYS_ADMIN) ||
     !ns_capable(cgroup_ns->user_ns, CAP_SYS_ADMIN))
  return -EPERM;


 if (cgroup_ns == nsproxy->cgroup_ns)
  return 0;

 get_cgroup_ns(cgroup_ns);
 put_cgroup_ns(nsproxy->cgroup_ns);
 nsproxy->cgroup_ns = cgroup_ns;

 return 0;
}

static struct ns_common *cgroupns_get(struct task_struct *task)
{
 struct cgroup_namespace *ns = NULL;
 struct nsproxy *nsproxy;

 task_lock(task);
 nsproxy = task->nsproxy;
 if (nsproxy) {
  ns = nsproxy->cgroup_ns;
  get_cgroup_ns(ns);
 }
 task_unlock(task);

 return ns ? &ns->ns : NULL;
}

static void cgroupns_put(struct ns_common *ns)
{
 put_cgroup_ns(to_cg_ns(ns));
}

const struct proc_ns_operations cgroupns_operations = {
 .name = "cgroup",
 .type = CLONE_NEWCGROUP,
 .get = cgroupns_get,
 .put = cgroupns_put,
 .install = cgroupns_install,
};

static __init int cgroup_namespaces_init(void)
{
 return 0;
}
subsys_initcall(cgroup_namespaces_init);

static struct cgroup_subsys_state *
debug_css_alloc(struct cgroup_subsys_state *parent_css)
{
 struct cgroup_subsys_state *css = kzalloc(sizeof(*css), GFP_KERNEL);

 if (!css)
  return ERR_PTR(-ENOMEM);

 return css;
}

static void debug_css_free(struct cgroup_subsys_state *css)
{
 kfree(css);
}

static u64 debug_taskcount_read(struct cgroup_subsys_state *css,
    struct cftype *cft)
{
 return cgroup_task_count(css->cgroup);
}

static u64 current_css_set_read(struct cgroup_subsys_state *css,
    struct cftype *cft)
{
 return (u64)(unsigned long)current->cgroups;
}

static u64 current_css_set_refcount_read(struct cgroup_subsys_state *css,
      struct cftype *cft)
{
 u64 count;

 rcu_read_lock();
 count = atomic_read(&task_css_set(current)->refcount);
 rcu_read_unlock();
 return count;
}

static int current_css_set_cg_links_read(struct seq_file *seq, void *v)
{
 struct cgrp_cset_link *link;
 struct css_set *cset;
 char *name_buf;

 name_buf = kmalloc(NAME_MAX + 1, GFP_KERNEL);
 if (!name_buf)
  return -ENOMEM;

 spin_lock_irq(&css_set_lock);
 rcu_read_lock();
 cset = rcu_dereference(current->cgroups);
 list_for_each_entry(link, &cset->cgrp_links, cgrp_link) {
  struct cgroup *c = link->cgrp;

  cgroup_name(c, name_buf, NAME_MAX + 1);
  seq_printf(seq, "Root %d group %s\n",
      c->root->hierarchy_id, name_buf);
 }
 rcu_read_unlock();
 spin_unlock_irq(&css_set_lock);
 kfree(name_buf);
 return 0;
}

static int cgroup_css_links_read(struct seq_file *seq, void *v)
{
 struct cgroup_subsys_state *css = seq_css(seq);
 struct cgrp_cset_link *link;

 spin_lock_irq(&css_set_lock);
 list_for_each_entry(link, &css->cgroup->cset_links, cset_link) {
  struct css_set *cset = link->cset;
  struct task_struct *task;
  int count = 0;

  seq_printf(seq, "css_set %p\n", cset);

  list_for_each_entry(task, &cset->tasks, cg_list) {
   if (count++ > MAX_TASKS_SHOWN_PER_CSS)
    goto overflow;
   seq_printf(seq, "  task %d\n", task_pid_vnr(task));
  }

  list_for_each_entry(task, &cset->mg_tasks, cg_list) {
   if (count++ > MAX_TASKS_SHOWN_PER_CSS)
    goto overflow;
   seq_printf(seq, "  task %d\n", task_pid_vnr(task));
  }
  continue;
 overflow:
  seq_puts(seq, "  ...\n");
 }
 spin_unlock_irq(&css_set_lock);
 return 0;
}

static u64 releasable_read(struct cgroup_subsys_state *css, struct cftype *cft)
{
 return (!cgroup_is_populated(css->cgroup) &&
  !css_has_online_children(&css->cgroup->self));
}

static struct cftype debug_files[] = {
 {
  .name = "taskcount",
  .read_u64 = debug_taskcount_read,
 },

 {
  .name = "current_css_set",
  .read_u64 = current_css_set_read,
 },

 {
  .name = "current_css_set_refcount",
  .read_u64 = current_css_set_refcount_read,
 },

 {
  .name = "current_css_set_cg_links",
  .seq_show = current_css_set_cg_links_read,
 },

 {
  .name = "cgroup_css_links",
  .seq_show = cgroup_css_links_read,
 },

 {
  .name = "releasable",
  .read_u64 = releasable_read,
 },

 { }
};

struct cgroup_subsys debug_cgrp_subsys = {
 .css_alloc = debug_css_alloc,
 .css_free = debug_css_free,
 .legacy_cftypes = debug_files,
};
enum freezer_state_flags {
 CGROUP_FREEZER_ONLINE = (1 << 0),
 CGROUP_FREEZING_SELF = (1 << 1),
 CGROUP_FREEZING_PARENT = (1 << 2),
 CGROUP_FROZEN = (1 << 3),


 CGROUP_FREEZING = CGROUP_FREEZING_SELF | CGROUP_FREEZING_PARENT,
};

struct freezer {
 struct cgroup_subsys_state css;
 unsigned int state;
};

static DEFINE_MUTEX(freezer_mutex);

static inline struct freezer *css_freezer(struct cgroup_subsys_state *css)
{
 return css ? container_of(css, struct freezer, css) : NULL;
}

static inline struct freezer *task_freezer(struct task_struct *task)
{
 return css_freezer(task_css(task, freezer_cgrp_id));
}

static struct freezer *parent_freezer(struct freezer *freezer)
{
 return css_freezer(freezer->css.parent);
}

bool cgroup_freezing(struct task_struct *task)
{
 bool ret;

 rcu_read_lock();
 ret = task_freezer(task)->state & CGROUP_FREEZING;
 rcu_read_unlock();

 return ret;
}

static const char *freezer_state_strs(unsigned int state)
{
 if (state & CGROUP_FROZEN)
  return "FROZEN";
 if (state & CGROUP_FREEZING)
  return "FREEZING";
 return "THAWED";
};

static struct cgroup_subsys_state *
freezer_css_alloc(struct cgroup_subsys_state *parent_css)
{
 struct freezer *freezer;

 freezer = kzalloc(sizeof(struct freezer), GFP_KERNEL);
 if (!freezer)
  return ERR_PTR(-ENOMEM);

 return &freezer->css;
}
static int freezer_css_online(struct cgroup_subsys_state *css)
{
 struct freezer *freezer = css_freezer(css);
 struct freezer *parent = parent_freezer(freezer);

 mutex_lock(&freezer_mutex);

 freezer->state |= CGROUP_FREEZER_ONLINE;

 if (parent && (parent->state & CGROUP_FREEZING)) {
  freezer->state |= CGROUP_FREEZING_PARENT | CGROUP_FROZEN;
  atomic_inc(&system_freezing_cnt);
 }

 mutex_unlock(&freezer_mutex);
 return 0;
}
static void freezer_css_offline(struct cgroup_subsys_state *css)
{
 struct freezer *freezer = css_freezer(css);

 mutex_lock(&freezer_mutex);

 if (freezer->state & CGROUP_FREEZING)
  atomic_dec(&system_freezing_cnt);

 freezer->state = 0;

 mutex_unlock(&freezer_mutex);
}

static void freezer_css_free(struct cgroup_subsys_state *css)
{
 kfree(css_freezer(css));
}
static void freezer_attach(struct cgroup_taskset *tset)
{
 struct task_struct *task;
 struct cgroup_subsys_state *new_css;

 mutex_lock(&freezer_mutex);
 cgroup_taskset_for_each(task, new_css, tset) {
  struct freezer *freezer = css_freezer(new_css);

  if (!(freezer->state & CGROUP_FREEZING)) {
   __thaw_task(task);
  } else {
   freeze_task(task);

   while (freezer && (freezer->state & CGROUP_FROZEN)) {
    freezer->state &= ~CGROUP_FROZEN;
    freezer = parent_freezer(freezer);
   }
  }
 }

 mutex_unlock(&freezer_mutex);
}
static void freezer_fork(struct task_struct *task)
{
 struct freezer *freezer;
 if (task_css_is_root(task, freezer_cgrp_id))
  return;

 mutex_lock(&freezer_mutex);
 rcu_read_lock();

 freezer = task_freezer(task);
 if (freezer->state & CGROUP_FREEZING)
  freeze_task(task);

 rcu_read_unlock();
 mutex_unlock(&freezer_mutex);
}
static void update_if_frozen(struct cgroup_subsys_state *css)
{
 struct freezer *freezer = css_freezer(css);
 struct cgroup_subsys_state *pos;
 struct css_task_iter it;
 struct task_struct *task;

 lockdep_assert_held(&freezer_mutex);

 if (!(freezer->state & CGROUP_FREEZING) ||
     (freezer->state & CGROUP_FROZEN))
  return;


 rcu_read_lock();
 css_for_each_child(pos, css) {
  struct freezer *child = css_freezer(pos);

  if ((child->state & CGROUP_FREEZER_ONLINE) &&
      !(child->state & CGROUP_FROZEN)) {
   rcu_read_unlock();
   return;
  }
 }
 rcu_read_unlock();


 css_task_iter_start(css, &it);

 while ((task = css_task_iter_next(&it))) {
  if (freezing(task)) {






   if (!frozen(task) && !freezer_should_skip(task))
    goto out_iter_end;
  }
 }

 freezer->state |= CGROUP_FROZEN;
out_iter_end:
 css_task_iter_end(&it);
}

static int freezer_read(struct seq_file *m, void *v)
{
 struct cgroup_subsys_state *css = seq_css(m), *pos;

 mutex_lock(&freezer_mutex);
 rcu_read_lock();


 css_for_each_descendant_post(pos, css) {
  if (!css_tryget_online(pos))
   continue;
  rcu_read_unlock();

  update_if_frozen(pos);

  rcu_read_lock();
  css_put(pos);
 }

 rcu_read_unlock();
 mutex_unlock(&freezer_mutex);

 seq_puts(m, freezer_state_strs(css_freezer(css)->state));
 seq_putc(m, '\n');
 return 0;
}

static void freeze_cgroup(struct freezer *freezer)
{
 struct css_task_iter it;
 struct task_struct *task;

 css_task_iter_start(&freezer->css, &it);
 while ((task = css_task_iter_next(&it)))
  freeze_task(task);
 css_task_iter_end(&it);
}

static void unfreeze_cgroup(struct freezer *freezer)
{
 struct css_task_iter it;
 struct task_struct *task;

 css_task_iter_start(&freezer->css, &it);
 while ((task = css_task_iter_next(&it)))
  __thaw_task(task);
 css_task_iter_end(&it);
}
static void freezer_apply_state(struct freezer *freezer, bool freeze,
    unsigned int state)
{

 lockdep_assert_held(&freezer_mutex);

 if (!(freezer->state & CGROUP_FREEZER_ONLINE))
  return;

 if (freeze) {
  if (!(freezer->state & CGROUP_FREEZING))
   atomic_inc(&system_freezing_cnt);
  freezer->state |= state;
  freeze_cgroup(freezer);
 } else {
  bool was_freezing = freezer->state & CGROUP_FREEZING;

  freezer->state &= ~state;

  if (!(freezer->state & CGROUP_FREEZING)) {
   if (was_freezing)
    atomic_dec(&system_freezing_cnt);
   freezer->state &= ~CGROUP_FROZEN;
   unfreeze_cgroup(freezer);
  }
 }
}
static void freezer_change_state(struct freezer *freezer, bool freeze)
{
 struct cgroup_subsys_state *pos;






 mutex_lock(&freezer_mutex);
 rcu_read_lock();
 css_for_each_descendant_pre(pos, &freezer->css) {
  struct freezer *pos_f = css_freezer(pos);
  struct freezer *parent = parent_freezer(pos_f);

  if (!css_tryget_online(pos))
   continue;
  rcu_read_unlock();

  if (pos_f == freezer)
   freezer_apply_state(pos_f, freeze,
         CGROUP_FREEZING_SELF);
  else
   freezer_apply_state(pos_f,
         parent->state & CGROUP_FREEZING,
         CGROUP_FREEZING_PARENT);

  rcu_read_lock();
  css_put(pos);
 }
 rcu_read_unlock();
 mutex_unlock(&freezer_mutex);
}

static ssize_t freezer_write(struct kernfs_open_file *of,
        char *buf, size_t nbytes, loff_t off)
{
 bool freeze;

 buf = strstrip(buf);

 if (strcmp(buf, freezer_state_strs(0)) == 0)
  freeze = false;
 else if (strcmp(buf, freezer_state_strs(CGROUP_FROZEN)) == 0)
  freeze = true;
 else
  return -EINVAL;

 freezer_change_state(css_freezer(of_css(of)), freeze);
 return nbytes;
}

static u64 freezer_self_freezing_read(struct cgroup_subsys_state *css,
          struct cftype *cft)
{
 struct freezer *freezer = css_freezer(css);

 return (bool)(freezer->state & CGROUP_FREEZING_SELF);
}

static u64 freezer_parent_freezing_read(struct cgroup_subsys_state *css,
     struct cftype *cft)
{
 struct freezer *freezer = css_freezer(css);

 return (bool)(freezer->state & CGROUP_FREEZING_PARENT);
}

static struct cftype files[] = {
 {
  .name = "state",
  .flags = CFTYPE_NOT_ON_ROOT,
  .seq_show = freezer_read,
  .write = freezer_write,
 },
 {
  .name = "self_freezing",
  .flags = CFTYPE_NOT_ON_ROOT,
  .read_u64 = freezer_self_freezing_read,
 },
 {
  .name = "parent_freezing",
  .flags = CFTYPE_NOT_ON_ROOT,
  .read_u64 = freezer_parent_freezing_read,
 },
 { }
};

struct cgroup_subsys freezer_cgrp_subsys = {
 .css_alloc = freezer_css_alloc,
 .css_online = freezer_css_online,
 .css_offline = freezer_css_offline,
 .css_free = freezer_css_free,
 .attach = freezer_attach,
 .fork = freezer_fork,
 .legacy_cftypes = files,
};


struct pids_cgroup {
 struct cgroup_subsys_state css;





 atomic64_t counter;
 int64_t limit;
};

static struct pids_cgroup *css_pids(struct cgroup_subsys_state *css)
{
 return container_of(css, struct pids_cgroup, css);
}

static struct pids_cgroup *parent_pids(struct pids_cgroup *pids)
{
 return css_pids(pids->css.parent);
}

static struct cgroup_subsys_state *
pids_css_alloc(struct cgroup_subsys_state *parent)
{
 struct pids_cgroup *pids;

 pids = kzalloc(sizeof(struct pids_cgroup), GFP_KERNEL);
 if (!pids)
  return ERR_PTR(-ENOMEM);

 pids->limit = PIDS_MAX;
 atomic64_set(&pids->counter, 0);
 return &pids->css;
}

static void pids_css_free(struct cgroup_subsys_state *css)
{
 kfree(css_pids(css));
}
static void pids_cancel(struct pids_cgroup *pids, int num)
{




 WARN_ON_ONCE(atomic64_add_negative(-num, &pids->counter));
}






static void pids_uncharge(struct pids_cgroup *pids, int num)
{
 struct pids_cgroup *p;

 for (p = pids; parent_pids(p); p = parent_pids(p))
  pids_cancel(p, num);
}
static void pids_charge(struct pids_cgroup *pids, int num)
{
 struct pids_cgroup *p;

 for (p = pids; parent_pids(p); p = parent_pids(p))
  atomic64_add(num, &p->counter);
}
static int pids_try_charge(struct pids_cgroup *pids, int num)
{
 struct pids_cgroup *p, *q;

 for (p = pids; parent_pids(p); p = parent_pids(p)) {
  int64_t new = atomic64_add_return(num, &p->counter);






  if (new > p->limit)
   goto revert;
 }

 return 0;

revert:
 for (q = pids; q != p; q = parent_pids(q))
  pids_cancel(q, num);
 pids_cancel(p, num);

 return -EAGAIN;
}

static int pids_can_attach(struct cgroup_taskset *tset)
{
 struct task_struct *task;
 struct cgroup_subsys_state *dst_css;

 cgroup_taskset_for_each(task, dst_css, tset) {
  struct pids_cgroup *pids = css_pids(dst_css);
  struct cgroup_subsys_state *old_css;
  struct pids_cgroup *old_pids;






  old_css = task_css(task, pids_cgrp_id);
  old_pids = css_pids(old_css);

  pids_charge(pids, 1);
  pids_uncharge(old_pids, 1);
 }

 return 0;
}

static void pids_cancel_attach(struct cgroup_taskset *tset)
{
 struct task_struct *task;
 struct cgroup_subsys_state *dst_css;

 cgroup_taskset_for_each(task, dst_css, tset) {
  struct pids_cgroup *pids = css_pids(dst_css);
  struct cgroup_subsys_state *old_css;
  struct pids_cgroup *old_pids;

  old_css = task_css(task, pids_cgrp_id);
  old_pids = css_pids(old_css);

  pids_charge(old_pids, 1);
  pids_uncharge(pids, 1);
 }
}





static int pids_can_fork(struct task_struct *task)
{
 struct cgroup_subsys_state *css;
 struct pids_cgroup *pids;

 css = task_css_check(current, pids_cgrp_id, true);
 pids = css_pids(css);
 return pids_try_charge(pids, 1);
}

static void pids_cancel_fork(struct task_struct *task)
{
 struct cgroup_subsys_state *css;
 struct pids_cgroup *pids;

 css = task_css_check(current, pids_cgrp_id, true);
 pids = css_pids(css);
 pids_uncharge(pids, 1);
}

static void pids_free(struct task_struct *task)
{
 struct pids_cgroup *pids = css_pids(task_css(task, pids_cgrp_id));

 pids_uncharge(pids, 1);
}

static ssize_t pids_max_write(struct kernfs_open_file *of, char *buf,
         size_t nbytes, loff_t off)
{
 struct cgroup_subsys_state *css = of_css(of);
 struct pids_cgroup *pids = css_pids(css);
 int64_t limit;
 int err;

 buf = strstrip(buf);
 if (!strcmp(buf, PIDS_MAX_STR)) {
  limit = PIDS_MAX;
  goto set_limit;
 }

 err = kstrtoll(buf, 0, &limit);
 if (err)
  return err;

 if (limit < 0 || limit >= PIDS_MAX)
  return -EINVAL;

set_limit:




 pids->limit = limit;
 return nbytes;
}

static int pids_max_show(struct seq_file *sf, void *v)
{
 struct cgroup_subsys_state *css = seq_css(sf);
 struct pids_cgroup *pids = css_pids(css);
 int64_t limit = pids->limit;

 if (limit >= PIDS_MAX)
  seq_printf(sf, "%s\n", PIDS_MAX_STR);
 else
  seq_printf(sf, "%lld\n", limit);

 return 0;
}

static s64 pids_current_read(struct cgroup_subsys_state *css,
        struct cftype *cft)
{
 struct pids_cgroup *pids = css_pids(css);

 return atomic64_read(&pids->counter);
}

static struct cftype pids_files[] = {
 {
  .name = "max",
  .write = pids_max_write,
  .seq_show = pids_max_show,
  .flags = CFTYPE_NOT_ON_ROOT,
 },
 {
  .name = "current",
  .read_s64 = pids_current_read,
  .flags = CFTYPE_NOT_ON_ROOT,
 },
 { }
};

struct cgroup_subsys pids_cgrp_subsys = {
 .css_alloc = pids_css_alloc,
 .css_free = pids_css_free,
 .can_attach = pids_can_attach,
 .cancel_attach = pids_cancel_attach,
 .can_fork = pids_can_fork,
 .cancel_fork = pids_cancel_fork,
 .free = pids_free,
 .legacy_cftypes = pids_files,
 .dfl_cftypes = pids_files,
};



static irqreturn_t bad_chained_irq(int irq, void *dev_id)
{
 WARN_ONCE(1, "Chained irq %d should not call an action\n", irq);
 return IRQ_NONE;
}





struct irqaction chained_action = {
 .handler = bad_chained_irq,
};






int irq_set_chip(unsigned int irq, struct irq_chip *chip)
{
 unsigned long flags;
 struct irq_desc *desc = irq_get_desc_lock(irq, &flags, 0);

 if (!desc)
  return -EINVAL;

 if (!chip)
  chip = &no_irq_chip;

 desc->irq_data.chip = chip;
 irq_put_desc_unlock(desc, flags);




 irq_mark_irq(irq);
 return 0;
}
EXPORT_SYMBOL(irq_set_chip);






int irq_set_irq_type(unsigned int irq, unsigned int type)
{
 unsigned long flags;
 struct irq_desc *desc = irq_get_desc_buslock(irq, &flags, IRQ_GET_DESC_CHECK_GLOBAL);
 int ret = 0;

 if (!desc)
  return -EINVAL;

 type &= IRQ_TYPE_SENSE_MASK;
 ret = __irq_set_trigger(desc, type);
 irq_put_desc_busunlock(desc, flags);
 return ret;
}
EXPORT_SYMBOL(irq_set_irq_type);
int irq_set_handler_data(unsigned int irq, void *data)
{
 unsigned long flags;
 struct irq_desc *desc = irq_get_desc_lock(irq, &flags, 0);

 if (!desc)
  return -EINVAL;
 desc->irq_common_data.handler_data = data;
 irq_put_desc_unlock(desc, flags);
 return 0;
}
EXPORT_SYMBOL(irq_set_handler_data);
int irq_set_msi_desc_off(unsigned int irq_base, unsigned int irq_offset,
    struct msi_desc *entry)
{
 unsigned long flags;
 struct irq_desc *desc = irq_get_desc_lock(irq_base + irq_offset, &flags, IRQ_GET_DESC_CHECK_GLOBAL);

 if (!desc)
  return -EINVAL;
 desc->irq_common_data.msi_desc = entry;
 if (entry && !irq_offset)
  entry->irq = irq_base;
 irq_put_desc_unlock(desc, flags);
 return 0;
}
int irq_set_msi_desc(unsigned int irq, struct msi_desc *entry)
{
 return irq_set_msi_desc_off(irq, 0, entry);
}
int irq_set_chip_data(unsigned int irq, void *data)
{
 unsigned long flags;
 struct irq_desc *desc = irq_get_desc_lock(irq, &flags, 0);

 if (!desc)
  return -EINVAL;
 desc->irq_data.chip_data = data;
 irq_put_desc_unlock(desc, flags);
 return 0;
}
EXPORT_SYMBOL(irq_set_chip_data);

struct irq_data *irq_get_irq_data(unsigned int irq)
{
 struct irq_desc *desc = irq_to_desc(irq);

 return desc ? &desc->irq_data : NULL;
}
EXPORT_SYMBOL_GPL(irq_get_irq_data);

static void irq_state_clr_disabled(struct irq_desc *desc)
{
 irqd_clear(&desc->irq_data, IRQD_IRQ_DISABLED);
}

static void irq_state_set_disabled(struct irq_desc *desc)
{
 irqd_set(&desc->irq_data, IRQD_IRQ_DISABLED);
}

static void irq_state_clr_masked(struct irq_desc *desc)
{
 irqd_clear(&desc->irq_data, IRQD_IRQ_MASKED);
}

static void irq_state_set_masked(struct irq_desc *desc)
{
 irqd_set(&desc->irq_data, IRQD_IRQ_MASKED);
}

int irq_startup(struct irq_desc *desc, bool resend)
{
 int ret = 0;

 irq_state_clr_disabled(desc);
 desc->depth = 0;

 irq_domain_activate_irq(&desc->irq_data);
 if (desc->irq_data.chip->irq_startup) {
  ret = desc->irq_data.chip->irq_startup(&desc->irq_data);
  irq_state_clr_masked(desc);
 } else {
  irq_enable(desc);
 }
 if (resend)
  check_irq_resend(desc);
 return ret;
}

void irq_shutdown(struct irq_desc *desc)
{
 irq_state_set_disabled(desc);
 desc->depth = 1;
 if (desc->irq_data.chip->irq_shutdown)
  desc->irq_data.chip->irq_shutdown(&desc->irq_data);
 else if (desc->irq_data.chip->irq_disable)
  desc->irq_data.chip->irq_disable(&desc->irq_data);
 else
  desc->irq_data.chip->irq_mask(&desc->irq_data);
 irq_domain_deactivate_irq(&desc->irq_data);
 irq_state_set_masked(desc);
}

void irq_enable(struct irq_desc *desc)
{
 irq_state_clr_disabled(desc);
 if (desc->irq_data.chip->irq_enable)
  desc->irq_data.chip->irq_enable(&desc->irq_data);
 else
  desc->irq_data.chip->irq_unmask(&desc->irq_data);
 irq_state_clr_masked(desc);
}
void irq_disable(struct irq_desc *desc)
{
 irq_state_set_disabled(desc);
 if (desc->irq_data.chip->irq_disable) {
  desc->irq_data.chip->irq_disable(&desc->irq_data);
  irq_state_set_masked(desc);
 } else if (irq_settings_disable_unlazy(desc)) {
  mask_irq(desc);
 }
}

void irq_percpu_enable(struct irq_desc *desc, unsigned int cpu)
{
 if (desc->irq_data.chip->irq_enable)
  desc->irq_data.chip->irq_enable(&desc->irq_data);
 else
  desc->irq_data.chip->irq_unmask(&desc->irq_data);
 cpumask_set_cpu(cpu, desc->percpu_enabled);
}

void irq_percpu_disable(struct irq_desc *desc, unsigned int cpu)
{
 if (desc->irq_data.chip->irq_disable)
  desc->irq_data.chip->irq_disable(&desc->irq_data);
 else
  desc->irq_data.chip->irq_mask(&desc->irq_data);
 cpumask_clear_cpu(cpu, desc->percpu_enabled);
}

static inline void mask_ack_irq(struct irq_desc *desc)
{
 if (desc->irq_data.chip->irq_mask_ack)
  desc->irq_data.chip->irq_mask_ack(&desc->irq_data);
 else {
  desc->irq_data.chip->irq_mask(&desc->irq_data);
  if (desc->irq_data.chip->irq_ack)
   desc->irq_data.chip->irq_ack(&desc->irq_data);
 }
 irq_state_set_masked(desc);
}

void mask_irq(struct irq_desc *desc)
{
 if (desc->irq_data.chip->irq_mask) {
  desc->irq_data.chip->irq_mask(&desc->irq_data);
  irq_state_set_masked(desc);
 }
}

void unmask_irq(struct irq_desc *desc)
{
 if (desc->irq_data.chip->irq_unmask) {
  desc->irq_data.chip->irq_unmask(&desc->irq_data);
  irq_state_clr_masked(desc);
 }
}

void unmask_threaded_irq(struct irq_desc *desc)
{
 struct irq_chip *chip = desc->irq_data.chip;

 if (chip->flags & IRQCHIP_EOI_THREADED)
  chip->irq_eoi(&desc->irq_data);

 if (chip->irq_unmask) {
  chip->irq_unmask(&desc->irq_data);
  irq_state_clr_masked(desc);
 }
}
void handle_nested_irq(unsigned int irq)
{
 struct irq_desc *desc = irq_to_desc(irq);
 struct irqaction *action;
 irqreturn_t action_ret;

 might_sleep();

 raw_spin_lock_irq(&desc->lock);

 desc->istate &= ~(IRQS_REPLAY | IRQS_WAITING);

 action = desc->action;
 if (unlikely(!action || irqd_irq_disabled(&desc->irq_data))) {
  desc->istate |= IRQS_PENDING;
  goto out_unlock;
 }

 kstat_incr_irqs_this_cpu(desc);
 irqd_set(&desc->irq_data, IRQD_IRQ_INPROGRESS);
 raw_spin_unlock_irq(&desc->lock);

 action_ret = action->thread_fn(action->irq, action->dev_id);
 if (!noirqdebug)
  note_interrupt(desc, action_ret);

 raw_spin_lock_irq(&desc->lock);
 irqd_clear(&desc->irq_data, IRQD_IRQ_INPROGRESS);

out_unlock:
 raw_spin_unlock_irq(&desc->lock);
}
EXPORT_SYMBOL_GPL(handle_nested_irq);

static bool irq_check_poll(struct irq_desc *desc)
{
 if (!(desc->istate & IRQS_POLL_INPROGRESS))
  return false;
 return irq_wait_for_poll(desc);
}

static bool irq_may_run(struct irq_desc *desc)
{
 unsigned int mask = IRQD_IRQ_INPROGRESS | IRQD_WAKEUP_ARMED;





 if (!irqd_has_set(&desc->irq_data, mask))
  return true;






 if (irq_pm_check_wakeup(desc))
  return false;




 return irq_check_poll(desc);
}
void handle_simple_irq(struct irq_desc *desc)
{
 raw_spin_lock(&desc->lock);

 if (!irq_may_run(desc))
  goto out_unlock;

 desc->istate &= ~(IRQS_REPLAY | IRQS_WAITING);

 if (unlikely(!desc->action || irqd_irq_disabled(&desc->irq_data))) {
  desc->istate |= IRQS_PENDING;
  goto out_unlock;
 }

 kstat_incr_irqs_this_cpu(desc);
 handle_irq_event(desc);

out_unlock:
 raw_spin_unlock(&desc->lock);
}
EXPORT_SYMBOL_GPL(handle_simple_irq);





static void cond_unmask_irq(struct irq_desc *desc)
{







 if (!irqd_irq_disabled(&desc->irq_data) &&
     irqd_irq_masked(&desc->irq_data) && !desc->threads_oneshot)
  unmask_irq(desc);
}
void handle_level_irq(struct irq_desc *desc)
{
 raw_spin_lock(&desc->lock);
 mask_ack_irq(desc);

 if (!irq_may_run(desc))
  goto out_unlock;

 desc->istate &= ~(IRQS_REPLAY | IRQS_WAITING);





 if (unlikely(!desc->action || irqd_irq_disabled(&desc->irq_data))) {
  desc->istate |= IRQS_PENDING;
  goto out_unlock;
 }

 kstat_incr_irqs_this_cpu(desc);
 handle_irq_event(desc);

 cond_unmask_irq(desc);

out_unlock:
 raw_spin_unlock(&desc->lock);
}
EXPORT_SYMBOL_GPL(handle_level_irq);

static inline void preflow_handler(struct irq_desc *desc)
{
 if (desc->preflow_handler)
  desc->preflow_handler(&desc->irq_data);
}
static inline void preflow_handler(struct irq_desc *desc) { }

static void cond_unmask_eoi_irq(struct irq_desc *desc, struct irq_chip *chip)
{
 if (!(desc->istate & IRQS_ONESHOT)) {
  chip->irq_eoi(&desc->irq_data);
  return;
 }






 if (!irqd_irq_disabled(&desc->irq_data) &&
     irqd_irq_masked(&desc->irq_data) && !desc->threads_oneshot) {
  chip->irq_eoi(&desc->irq_data);
  unmask_irq(desc);
 } else if (!(chip->flags & IRQCHIP_EOI_THREADED)) {
  chip->irq_eoi(&desc->irq_data);
 }
}
void handle_fasteoi_irq(struct irq_desc *desc)
{
 struct irq_chip *chip = desc->irq_data.chip;

 raw_spin_lock(&desc->lock);

 if (!irq_may_run(desc))
  goto out;

 desc->istate &= ~(IRQS_REPLAY | IRQS_WAITING);





 if (unlikely(!desc->action || irqd_irq_disabled(&desc->irq_data))) {
  desc->istate |= IRQS_PENDING;
  mask_irq(desc);
  goto out;
 }

 kstat_incr_irqs_this_cpu(desc);
 if (desc->istate & IRQS_ONESHOT)
  mask_irq(desc);

 preflow_handler(desc);
 handle_irq_event(desc);

 cond_unmask_eoi_irq(desc, chip);

 raw_spin_unlock(&desc->lock);
 return;
out:
 if (!(chip->flags & IRQCHIP_EOI_IF_HANDLED))
  chip->irq_eoi(&desc->irq_data);
 raw_spin_unlock(&desc->lock);
}
EXPORT_SYMBOL_GPL(handle_fasteoi_irq);
void handle_edge_irq(struct irq_desc *desc)
{
 raw_spin_lock(&desc->lock);

 desc->istate &= ~(IRQS_REPLAY | IRQS_WAITING);

 if (!irq_may_run(desc)) {
  desc->istate |= IRQS_PENDING;
  mask_ack_irq(desc);
  goto out_unlock;
 }





 if (irqd_irq_disabled(&desc->irq_data) || !desc->action) {
  desc->istate |= IRQS_PENDING;
  mask_ack_irq(desc);
  goto out_unlock;
 }

 kstat_incr_irqs_this_cpu(desc);


 desc->irq_data.chip->irq_ack(&desc->irq_data);

 do {
  if (unlikely(!desc->action)) {
   mask_irq(desc);
   goto out_unlock;
  }






  if (unlikely(desc->istate & IRQS_PENDING)) {
   if (!irqd_irq_disabled(&desc->irq_data) &&
       irqd_irq_masked(&desc->irq_data))
    unmask_irq(desc);
  }

  handle_irq_event(desc);

 } while ((desc->istate & IRQS_PENDING) &&
   !irqd_irq_disabled(&desc->irq_data));

out_unlock:
 raw_spin_unlock(&desc->lock);
}
EXPORT_SYMBOL(handle_edge_irq);








void handle_edge_eoi_irq(struct irq_desc *desc)
{
 struct irq_chip *chip = irq_desc_get_chip(desc);

 raw_spin_lock(&desc->lock);

 desc->istate &= ~(IRQS_REPLAY | IRQS_WAITING);

 if (!irq_may_run(desc)) {
  desc->istate |= IRQS_PENDING;
  goto out_eoi;
 }





 if (irqd_irq_disabled(&desc->irq_data) || !desc->action) {
  desc->istate |= IRQS_PENDING;
  goto out_eoi;
 }

 kstat_incr_irqs_this_cpu(desc);

 do {
  if (unlikely(!desc->action))
   goto out_eoi;

  handle_irq_event(desc);

 } while ((desc->istate & IRQS_PENDING) &&
   !irqd_irq_disabled(&desc->irq_data));

out_eoi:
 chip->irq_eoi(&desc->irq_data);
 raw_spin_unlock(&desc->lock);
}







void handle_percpu_irq(struct irq_desc *desc)
{
 struct irq_chip *chip = irq_desc_get_chip(desc);

 kstat_incr_irqs_this_cpu(desc);

 if (chip->irq_ack)
  chip->irq_ack(&desc->irq_data);

 handle_irq_event_percpu(desc);

 if (chip->irq_eoi)
  chip->irq_eoi(&desc->irq_data);
}
void handle_percpu_devid_irq(struct irq_desc *desc)
{
 struct irq_chip *chip = irq_desc_get_chip(desc);
 struct irqaction *action = desc->action;
 void *dev_id = raw_cpu_ptr(action->percpu_dev_id);
 unsigned int irq = irq_desc_get_irq(desc);
 irqreturn_t res;

 kstat_incr_irqs_this_cpu(desc);

 if (chip->irq_ack)
  chip->irq_ack(&desc->irq_data);

 trace_irq_handler_entry(irq, action);
 res = action->handler(irq, dev_id);
 trace_irq_handler_exit(irq, action, res);

 if (chip->irq_eoi)
  chip->irq_eoi(&desc->irq_data);
}

void
__irq_do_set_handler(struct irq_desc *desc, irq_flow_handler_t handle,
       int is_chained, const char *name)
{
 if (!handle) {
  handle = handle_bad_irq;
 } else {
  struct irq_data *irq_data = &desc->irq_data;







  while (irq_data) {
   if (irq_data->chip != &no_irq_chip)
    break;





   if (WARN_ON(is_chained))
    return;

   irq_data = irq_data->parent_data;
  }
  if (WARN_ON(!irq_data || irq_data->chip == &no_irq_chip))
   return;
 }


 if (handle == handle_bad_irq) {
  if (desc->irq_data.chip != &no_irq_chip)
   mask_ack_irq(desc);
  irq_state_set_disabled(desc);
  if (is_chained)
   desc->action = NULL;
  desc->depth = 1;
 }
 desc->handle_irq = handle;
 desc->name = name;

 if (handle != handle_bad_irq && is_chained) {
  irq_settings_set_noprobe(desc);
  irq_settings_set_norequest(desc);
  irq_settings_set_nothread(desc);
  desc->action = &chained_action;
  irq_startup(desc, true);
 }
}

void
__irq_set_handler(unsigned int irq, irq_flow_handler_t handle, int is_chained,
    const char *name)
{
 unsigned long flags;
 struct irq_desc *desc = irq_get_desc_buslock(irq, &flags, 0);

 if (!desc)
  return;

 __irq_do_set_handler(desc, handle, is_chained, name);
 irq_put_desc_busunlock(desc, flags);
}
EXPORT_SYMBOL_GPL(__irq_set_handler);

void
irq_set_chained_handler_and_data(unsigned int irq, irq_flow_handler_t handle,
     void *data)
{
 unsigned long flags;
 struct irq_desc *desc = irq_get_desc_buslock(irq, &flags, 0);

 if (!desc)
  return;

 __irq_do_set_handler(desc, handle, 1, NULL);
 desc->irq_common_data.handler_data = data;

 irq_put_desc_busunlock(desc, flags);
}
EXPORT_SYMBOL_GPL(irq_set_chained_handler_and_data);

void
irq_set_chip_and_handler_name(unsigned int irq, struct irq_chip *chip,
         irq_flow_handler_t handle, const char *name)
{
 irq_set_chip(irq, chip);
 __irq_set_handler(irq, handle, 0, name);
}
EXPORT_SYMBOL_GPL(irq_set_chip_and_handler_name);

void irq_modify_status(unsigned int irq, unsigned long clr, unsigned long set)
{
 unsigned long flags;
 struct irq_desc *desc = irq_get_desc_lock(irq, &flags, 0);

 if (!desc)
  return;
 irq_settings_clr_and_set(desc, clr, set);

 irqd_clear(&desc->irq_data, IRQD_NO_BALANCING | IRQD_PER_CPU |
     IRQD_TRIGGER_MASK | IRQD_LEVEL | IRQD_MOVE_PCNTXT);
 if (irq_settings_has_no_balance_set(desc))
  irqd_set(&desc->irq_data, IRQD_NO_BALANCING);
 if (irq_settings_is_per_cpu(desc))
  irqd_set(&desc->irq_data, IRQD_PER_CPU);
 if (irq_settings_can_move_pcntxt(desc))
  irqd_set(&desc->irq_data, IRQD_MOVE_PCNTXT);
 if (irq_settings_is_level(desc))
  irqd_set(&desc->irq_data, IRQD_LEVEL);

 irqd_set(&desc->irq_data, irq_settings_get_trigger_mask(desc));

 irq_put_desc_unlock(desc, flags);
}
EXPORT_SYMBOL_GPL(irq_modify_status);







void irq_cpu_online(void)
{
 struct irq_desc *desc;
 struct irq_chip *chip;
 unsigned long flags;
 unsigned int irq;

 for_each_active_irq(irq) {
  desc = irq_to_desc(irq);
  if (!desc)
   continue;

  raw_spin_lock_irqsave(&desc->lock, flags);

  chip = irq_data_get_irq_chip(&desc->irq_data);
  if (chip && chip->irq_cpu_online &&
      (!(chip->flags & IRQCHIP_ONOFFLINE_ENABLED) ||
       !irqd_irq_disabled(&desc->irq_data)))
   chip->irq_cpu_online(&desc->irq_data);

  raw_spin_unlock_irqrestore(&desc->lock, flags);
 }
}







void irq_cpu_offline(void)
{
 struct irq_desc *desc;
 struct irq_chip *chip;
 unsigned long flags;
 unsigned int irq;

 for_each_active_irq(irq) {
  desc = irq_to_desc(irq);
  if (!desc)
   continue;

  raw_spin_lock_irqsave(&desc->lock, flags);

  chip = irq_data_get_irq_chip(&desc->irq_data);
  if (chip && chip->irq_cpu_offline &&
      (!(chip->flags & IRQCHIP_ONOFFLINE_ENABLED) ||
       !irqd_irq_disabled(&desc->irq_data)))
   chip->irq_cpu_offline(&desc->irq_data);

  raw_spin_unlock_irqrestore(&desc->lock, flags);
 }
}






void irq_chip_enable_parent(struct irq_data *data)
{
 data = data->parent_data;
 if (data->chip->irq_enable)
  data->chip->irq_enable(data);
 else
  data->chip->irq_unmask(data);
}






void irq_chip_disable_parent(struct irq_data *data)
{
 data = data->parent_data;
 if (data->chip->irq_disable)
  data->chip->irq_disable(data);
 else
  data->chip->irq_mask(data);
}





void irq_chip_ack_parent(struct irq_data *data)
{
 data = data->parent_data;
 data->chip->irq_ack(data);
}
EXPORT_SYMBOL_GPL(irq_chip_ack_parent);





void irq_chip_mask_parent(struct irq_data *data)
{
 data = data->parent_data;
 data->chip->irq_mask(data);
}
EXPORT_SYMBOL_GPL(irq_chip_mask_parent);





void irq_chip_unmask_parent(struct irq_data *data)
{
 data = data->parent_data;
 data->chip->irq_unmask(data);
}
EXPORT_SYMBOL_GPL(irq_chip_unmask_parent);





void irq_chip_eoi_parent(struct irq_data *data)
{
 data = data->parent_data;
 data->chip->irq_eoi(data);
}
EXPORT_SYMBOL_GPL(irq_chip_eoi_parent);
int irq_chip_set_affinity_parent(struct irq_data *data,
     const struct cpumask *dest, bool force)
{
 data = data->parent_data;
 if (data->chip->irq_set_affinity)
  return data->chip->irq_set_affinity(data, dest, force);

 return -ENOSYS;
}
int irq_chip_set_type_parent(struct irq_data *data, unsigned int type)
{
 data = data->parent_data;

 if (data->chip->irq_set_type)
  return data->chip->irq_set_type(data, type);

 return -ENOSYS;
}
EXPORT_SYMBOL_GPL(irq_chip_set_type_parent);
int irq_chip_retrigger_hierarchy(struct irq_data *data)
{
 for (data = data->parent_data; data; data = data->parent_data)
  if (data->chip && data->chip->irq_retrigger)
   return data->chip->irq_retrigger(data);

 return 0;
}






int irq_chip_set_vcpu_affinity_parent(struct irq_data *data, void *vcpu_info)
{
 data = data->parent_data;
 if (data->chip->irq_set_vcpu_affinity)
  return data->chip->irq_set_vcpu_affinity(data, vcpu_info);

 return -ENOSYS;
}
int irq_chip_set_wake_parent(struct irq_data *data, unsigned int on)
{
 data = data->parent_data;
 if (data->chip->irq_set_wake)
  return data->chip->irq_set_wake(data, on);

 return -ENOSYS;
}
int irq_chip_compose_msi_msg(struct irq_data *data, struct msi_msg *msg)
{
 struct irq_data *pos = NULL;

 for (; data; data = data->parent_data)
  if (data->chip && data->chip->irq_compose_msi_msg)
   pos = data;
 if (!pos)
  return -ENOSYS;

 pos->chip->irq_compose_msi_msg(pos, msg);

 return 0;
}



static LIST_HEAD(clockevent_devices);
static LIST_HEAD(clockevents_released);

static DEFINE_RAW_SPINLOCK(clockevents_lock);

static DEFINE_MUTEX(clockevents_mutex);

struct ce_unbind {
 struct clock_event_device *ce;
 int res;
};

static u64 cev_delta2ns(unsigned long latch, struct clock_event_device *evt,
   bool ismax)
{
 u64 clc = (u64) latch << evt->shift;
 u64 rnd;

 if (unlikely(!evt->mult)) {
  evt->mult = 1;
  WARN_ON(1);
 }
 rnd = (u64) evt->mult - 1;





 if ((clc >> evt->shift) != (u64)latch)
  clc = ~0ULL;
 if ((~0ULL - clc > rnd) &&
     (!ismax || evt->mult <= (1ULL << evt->shift)))
  clc += rnd;

 do_div(clc, evt->mult);


 return clc > 1000 ? clc : 1000;
}
u64 clockevent_delta2ns(unsigned long latch, struct clock_event_device *evt)
{
 return cev_delta2ns(latch, evt, false);
}
EXPORT_SYMBOL_GPL(clockevent_delta2ns);

static int __clockevents_switch_state(struct clock_event_device *dev,
          enum clock_event_state state)
{
 if (dev->features & CLOCK_EVT_FEAT_DUMMY)
  return 0;


 switch (state) {
 case CLOCK_EVT_STATE_DETACHED:


 case CLOCK_EVT_STATE_SHUTDOWN:
  if (dev->set_state_shutdown)
   return dev->set_state_shutdown(dev);
  return 0;

 case CLOCK_EVT_STATE_PERIODIC:

  if (!(dev->features & CLOCK_EVT_FEAT_PERIODIC))
   return -ENOSYS;
  if (dev->set_state_periodic)
   return dev->set_state_periodic(dev);
  return 0;

 case CLOCK_EVT_STATE_ONESHOT:

  if (!(dev->features & CLOCK_EVT_FEAT_ONESHOT))
   return -ENOSYS;
  if (dev->set_state_oneshot)
   return dev->set_state_oneshot(dev);
  return 0;

 case CLOCK_EVT_STATE_ONESHOT_STOPPED:

  if (WARN_ONCE(!clockevent_state_oneshot(dev),
         "Current state: %d\n",
         clockevent_get_state(dev)))
   return -EINVAL;

  if (dev->set_state_oneshot_stopped)
   return dev->set_state_oneshot_stopped(dev);
  else
   return -ENOSYS;

 default:
  return -ENOSYS;
 }
}
void clockevents_switch_state(struct clock_event_device *dev,
         enum clock_event_state state)
{
 if (clockevent_get_state(dev) != state) {
  if (__clockevents_switch_state(dev, state))
   return;

  clockevent_set_state(dev, state);





  if (clockevent_state_oneshot(dev)) {
   if (unlikely(!dev->mult)) {
    dev->mult = 1;
    WARN_ON(1);
   }
  }
 }
}





void clockevents_shutdown(struct clock_event_device *dev)
{
 clockevents_switch_state(dev, CLOCK_EVT_STATE_SHUTDOWN);
 dev->next_event.tv64 = KTIME_MAX;
}





int clockevents_tick_resume(struct clock_event_device *dev)
{
 int ret = 0;

 if (dev->tick_resume)
  ret = dev->tick_resume(dev);

 return ret;
}










static int clockevents_increase_min_delta(struct clock_event_device *dev)
{

 if (dev->min_delta_ns >= MIN_DELTA_LIMIT) {
  printk_deferred(KERN_WARNING
    "CE: Reprogramming failure. Giving up\n");
  dev->next_event.tv64 = KTIME_MAX;
  return -ETIME;
 }

 if (dev->min_delta_ns < 5000)
  dev->min_delta_ns = 5000;
 else
  dev->min_delta_ns += dev->min_delta_ns >> 1;

 if (dev->min_delta_ns > MIN_DELTA_LIMIT)
  dev->min_delta_ns = MIN_DELTA_LIMIT;

 printk_deferred(KERN_WARNING
   "CE: %s increased min_delta_ns to %llu nsec\n",
   dev->name ? dev->name : "?",
   (unsigned long long) dev->min_delta_ns);
 return 0;
}







static int clockevents_program_min_delta(struct clock_event_device *dev)
{
 unsigned long long clc;
 int64_t delta;
 int i;

 for (i = 0;;) {
  delta = dev->min_delta_ns;
  dev->next_event = ktime_add_ns(ktime_get(), delta);

  if (clockevent_state_shutdown(dev))
   return 0;

  dev->retries++;
  clc = ((unsigned long long) delta * dev->mult) >> dev->shift;
  if (dev->set_next_event((unsigned long) clc, dev) == 0)
   return 0;

  if (++i > 2) {





   if (clockevents_increase_min_delta(dev))
    return -ETIME;
   i = 0;
  }
 }
}








static int clockevents_program_min_delta(struct clock_event_device *dev)
{
 unsigned long long clc;
 int64_t delta;

 delta = dev->min_delta_ns;
 dev->next_event = ktime_add_ns(ktime_get(), delta);

 if (clockevent_state_shutdown(dev))
  return 0;

 dev->retries++;
 clc = ((unsigned long long) delta * dev->mult) >> dev->shift;
 return dev->set_next_event((unsigned long) clc, dev);
}

int clockevents_program_event(struct clock_event_device *dev, ktime_t expires,
         bool force)
{
 unsigned long long clc;
 int64_t delta;
 int rc;

 if (unlikely(expires.tv64 < 0)) {
  WARN_ON_ONCE(1);
  return -ETIME;
 }

 dev->next_event = expires;

 if (clockevent_state_shutdown(dev))
  return 0;


 WARN_ONCE(!clockevent_state_oneshot(dev), "Current state: %d\n",
    clockevent_get_state(dev));


 if (dev->features & CLOCK_EVT_FEAT_KTIME)
  return dev->set_next_ktime(expires, dev);

 delta = ktime_to_ns(ktime_sub(expires, ktime_get()));
 if (delta <= 0)
  return force ? clockevents_program_min_delta(dev) : -ETIME;

 delta = min(delta, (int64_t) dev->max_delta_ns);
 delta = max(delta, (int64_t) dev->min_delta_ns);

 clc = ((unsigned long long) delta * dev->mult) >> dev->shift;
 rc = dev->set_next_event((unsigned long) clc, dev);

 return (rc && force) ? clockevents_program_min_delta(dev) : rc;
}





static void clockevents_notify_released(void)
{
 struct clock_event_device *dev;

 while (!list_empty(&clockevents_released)) {
  dev = list_entry(clockevents_released.next,
     struct clock_event_device, list);
  list_del(&dev->list);
  list_add(&dev->list, &clockevent_devices);
  tick_check_new_device(dev);
 }
}




static int clockevents_replace(struct clock_event_device *ced)
{
 struct clock_event_device *dev, *newdev = NULL;

 list_for_each_entry(dev, &clockevent_devices, list) {
  if (dev == ced || !clockevent_state_detached(dev))
   continue;

  if (!tick_check_replacement(newdev, dev))
   continue;

  if (!try_module_get(dev->owner))
   continue;

  if (newdev)
   module_put(newdev->owner);
  newdev = dev;
 }
 if (newdev) {
  tick_install_replacement(newdev);
  list_del_init(&ced->list);
 }
 return newdev ? 0 : -EBUSY;
}




static int __clockevents_try_unbind(struct clock_event_device *ced, int cpu)
{

 if (clockevent_state_detached(ced)) {
  list_del_init(&ced->list);
  return 0;
 }

 return ced == per_cpu(tick_cpu_device, cpu).evtdev ? -EAGAIN : -EBUSY;
}




static void __clockevents_unbind(void *arg)
{
 struct ce_unbind *cu = arg;
 int res;

 raw_spin_lock(&clockevents_lock);
 res = __clockevents_try_unbind(cu->ce, smp_processor_id());
 if (res == -EAGAIN)
  res = clockevents_replace(cu->ce);
 cu->res = res;
 raw_spin_unlock(&clockevents_lock);
}





static int clockevents_unbind(struct clock_event_device *ced, int cpu)
{
 struct ce_unbind cu = { .ce = ced, .res = -ENODEV };

 smp_call_function_single(cpu, __clockevents_unbind, &cu, 1);
 return cu.res;
}




int clockevents_unbind_device(struct clock_event_device *ced, int cpu)
{
 int ret;

 mutex_lock(&clockevents_mutex);
 ret = clockevents_unbind(ced, cpu);
 mutex_unlock(&clockevents_mutex);
 return ret;
}
EXPORT_SYMBOL_GPL(clockevents_unbind_device);





void clockevents_register_device(struct clock_event_device *dev)
{
 unsigned long flags;


 clockevent_set_state(dev, CLOCK_EVT_STATE_DETACHED);

 if (!dev->cpumask) {
  WARN_ON(num_possible_cpus() > 1);
  dev->cpumask = cpumask_of(smp_processor_id());
 }

 raw_spin_lock_irqsave(&clockevents_lock, flags);

 list_add(&dev->list, &clockevent_devices);
 tick_check_new_device(dev);
 clockevents_notify_released();

 raw_spin_unlock_irqrestore(&clockevents_lock, flags);
}
EXPORT_SYMBOL_GPL(clockevents_register_device);

void clockevents_config(struct clock_event_device *dev, u32 freq)
{
 u64 sec;

 if (!(dev->features & CLOCK_EVT_FEAT_ONESHOT))
  return;






 sec = dev->max_delta_ticks;
 do_div(sec, freq);
 if (!sec)
  sec = 1;
 else if (sec > 600 && dev->max_delta_ticks > UINT_MAX)
  sec = 600;

 clockevents_calc_mult_shift(dev, freq, sec);
 dev->min_delta_ns = cev_delta2ns(dev->min_delta_ticks, dev, false);
 dev->max_delta_ns = cev_delta2ns(dev->max_delta_ticks, dev, true);
}
void clockevents_config_and_register(struct clock_event_device *dev,
         u32 freq, unsigned long min_delta,
         unsigned long max_delta)
{
 dev->min_delta_ticks = min_delta;
 dev->max_delta_ticks = max_delta;
 clockevents_config(dev, freq);
 clockevents_register_device(dev);
}
EXPORT_SYMBOL_GPL(clockevents_config_and_register);

int __clockevents_update_freq(struct clock_event_device *dev, u32 freq)
{
 clockevents_config(dev, freq);

 if (clockevent_state_oneshot(dev))
  return clockevents_program_event(dev, dev->next_event, false);

 if (clockevent_state_periodic(dev))
  return __clockevents_switch_state(dev, CLOCK_EVT_STATE_PERIODIC);

 return 0;
}
int clockevents_update_freq(struct clock_event_device *dev, u32 freq)
{
 unsigned long flags;
 int ret;

 local_irq_save(flags);
 ret = tick_broadcast_update_freq(dev, freq);
 if (ret == -ENODEV)
  ret = __clockevents_update_freq(dev, freq);
 local_irq_restore(flags);
 return ret;
}




void clockevents_handle_noop(struct clock_event_device *dev)
{
}
void clockevents_exchange_device(struct clock_event_device *old,
     struct clock_event_device *new)
{




 if (old) {
  module_put(old->owner);
  clockevents_switch_state(old, CLOCK_EVT_STATE_DETACHED);
  list_del(&old->list);
  list_add(&old->list, &clockevents_released);
 }

 if (new) {
  BUG_ON(!clockevent_state_detached(new));
  clockevents_shutdown(new);
 }
}




void clockevents_suspend(void)
{
 struct clock_event_device *dev;

 list_for_each_entry_reverse(dev, &clockevent_devices, list)
  if (dev->suspend && !clockevent_state_detached(dev))
   dev->suspend(dev);
}




void clockevents_resume(void)
{
 struct clock_event_device *dev;

 list_for_each_entry(dev, &clockevent_devices, list)
  if (dev->resume && !clockevent_state_detached(dev))
   dev->resume(dev);
}




void tick_cleanup_dead_cpu(int cpu)
{
 struct clock_event_device *dev, *tmp;
 unsigned long flags;

 raw_spin_lock_irqsave(&clockevents_lock, flags);

 tick_shutdown_broadcast_oneshot(cpu);
 tick_shutdown_broadcast(cpu);
 tick_shutdown(cpu);




 list_for_each_entry_safe(dev, tmp, &clockevents_released, list)
  list_del(&dev->list);



 list_for_each_entry_safe(dev, tmp, &clockevent_devices, list) {
  if (cpumask_test_cpu(cpu, dev->cpumask) &&
      cpumask_weight(dev->cpumask) == 1 &&
      !tick_is_broadcast_device(dev)) {
   BUG_ON(!clockevent_state_detached(dev));
   list_del(&dev->list);
  }
 }
 raw_spin_unlock_irqrestore(&clockevents_lock, flags);
}

struct bus_type clockevents_subsys = {
 .name = "clockevents",
 .dev_name = "clockevent",
};

static DEFINE_PER_CPU(struct device, tick_percpu_dev);
static struct tick_device *tick_get_tick_dev(struct device *dev);

static ssize_t sysfs_show_current_tick_dev(struct device *dev,
        struct device_attribute *attr,
        char *buf)
{
 struct tick_device *td;
 ssize_t count = 0;

 raw_spin_lock_irq(&clockevents_lock);
 td = tick_get_tick_dev(dev);
 if (td && td->evtdev)
  count = snprintf(buf, PAGE_SIZE, "%s\n", td->evtdev->name);
 raw_spin_unlock_irq(&clockevents_lock);
 return count;
}
static DEVICE_ATTR(current_device, 0444, sysfs_show_current_tick_dev, NULL);


static ssize_t sysfs_unbind_tick_dev(struct device *dev,
         struct device_attribute *attr,
         const char *buf, size_t count)
{
 char name[CS_NAME_LEN];
 ssize_t ret = sysfs_get_uname(buf, name, count);
 struct clock_event_device *ce;

 if (ret < 0)
  return ret;

 ret = -ENODEV;
 mutex_lock(&clockevents_mutex);
 raw_spin_lock_irq(&clockevents_lock);
 list_for_each_entry(ce, &clockevent_devices, list) {
  if (!strcmp(ce->name, name)) {
   ret = __clockevents_try_unbind(ce, dev->id);
   break;
  }
 }
 raw_spin_unlock_irq(&clockevents_lock);



 if (ret == -EAGAIN)
  ret = clockevents_unbind(ce, dev->id);
 mutex_unlock(&clockevents_mutex);
 return ret ? ret : count;
}
static DEVICE_ATTR(unbind_device, 0200, NULL, sysfs_unbind_tick_dev);

static struct device tick_bc_dev = {
 .init_name = "broadcast",
 .id = 0,
 .bus = &clockevents_subsys,
};

static struct tick_device *tick_get_tick_dev(struct device *dev)
{
 return dev == &tick_bc_dev ? tick_get_broadcast_device() :
  &per_cpu(tick_cpu_device, dev->id);
}

static __init int tick_broadcast_init_sysfs(void)
{
 int err = device_register(&tick_bc_dev);

 if (!err)
  err = device_create_file(&tick_bc_dev, &dev_attr_current_device);
 return err;
}
static struct tick_device *tick_get_tick_dev(struct device *dev)
{
 return &per_cpu(tick_cpu_device, dev->id);
}
static inline int tick_broadcast_init_sysfs(void) { return 0; }

static int __init tick_init_sysfs(void)
{
 int cpu;

 for_each_possible_cpu(cpu) {
  struct device *dev = &per_cpu(tick_percpu_dev, cpu);
  int err;

  dev->id = cpu;
  dev->bus = &clockevents_subsys;
  err = device_register(dev);
  if (!err)
   err = device_create_file(dev, &dev_attr_current_device);
  if (!err)
   err = device_create_file(dev, &dev_attr_unbind_device);
  if (err)
   return err;
 }
 return tick_broadcast_init_sysfs();
}

static int __init clockevents_init_sysfs(void)
{
 int err = subsys_system_register(&clockevents_subsys, NULL);

 if (!err)
  err = tick_init_sysfs();
 return err;
}
device_initcall(clockevents_init_sysfs);


void
clocks_calc_mult_shift(u32 *mult, u32 *shift, u32 from, u32 to, u32 maxsec)
{
 u64 tmp;
 u32 sft, sftacc= 32;





 tmp = ((u64)maxsec * from) >> 32;
 while (tmp) {
  tmp >>=1;
  sftacc--;
 }





 for (sft = 32; sft > 0; sft--) {
  tmp = (u64) to << sft;
  tmp += from / 2;
  do_div(tmp, from);
  if ((tmp >> sftacc) == 0)
   break;
 }
 *mult = tmp;
 *shift = sft;
}
static struct clocksource *curr_clocksource;
static LIST_HEAD(clocksource_list);
static DEFINE_MUTEX(clocksource_mutex);
static char override_name[CS_NAME_LEN];
static int finished_booting;

static void clocksource_watchdog_work(struct work_struct *work);
static void clocksource_select(void);

static LIST_HEAD(watchdog_list);
static struct clocksource *watchdog;
static struct timer_list watchdog_timer;
static DECLARE_WORK(watchdog_work, clocksource_watchdog_work);
static DEFINE_SPINLOCK(watchdog_lock);
static int watchdog_running;
static atomic_t watchdog_reset_pending;

static int clocksource_watchdog_kthread(void *data);
static void __clocksource_change_rating(struct clocksource *cs, int rating);





static void clocksource_watchdog_work(struct work_struct *work)
{




 kthread_run(clocksource_watchdog_kthread, NULL, "kwatchdog");
}

static void __clocksource_unstable(struct clocksource *cs)
{
 cs->flags &= ~(CLOCK_SOURCE_VALID_FOR_HRES | CLOCK_SOURCE_WATCHDOG);
 cs->flags |= CLOCK_SOURCE_UNSTABLE;
 if (finished_booting)
  schedule_work(&watchdog_work);
}
void clocksource_mark_unstable(struct clocksource *cs)
{
 unsigned long flags;

 spin_lock_irqsave(&watchdog_lock, flags);
 if (!(cs->flags & CLOCK_SOURCE_UNSTABLE)) {
  if (list_empty(&cs->wd_list))
   list_add(&cs->wd_list, &watchdog_list);
  __clocksource_unstable(cs);
 }
 spin_unlock_irqrestore(&watchdog_lock, flags);
}

static void clocksource_watchdog(unsigned long data)
{
 struct clocksource *cs;
 cycle_t csnow, wdnow, cslast, wdlast, delta;
 int64_t wd_nsec, cs_nsec;
 int next_cpu, reset_pending;

 spin_lock(&watchdog_lock);
 if (!watchdog_running)
  goto out;

 reset_pending = atomic_read(&watchdog_reset_pending);

 list_for_each_entry(cs, &watchdog_list, wd_list) {


  if (cs->flags & CLOCK_SOURCE_UNSTABLE) {
   if (finished_booting)
    schedule_work(&watchdog_work);
   continue;
  }

  local_irq_disable();
  csnow = cs->read(cs);
  wdnow = watchdog->read(watchdog);
  local_irq_enable();


  if (!(cs->flags & CLOCK_SOURCE_WATCHDOG) ||
      atomic_read(&watchdog_reset_pending)) {
   cs->flags |= CLOCK_SOURCE_WATCHDOG;
   cs->wd_last = wdnow;
   cs->cs_last = csnow;
   continue;
  }

  delta = clocksource_delta(wdnow, cs->wd_last, watchdog->mask);
  wd_nsec = clocksource_cyc2ns(delta, watchdog->mult,
          watchdog->shift);

  delta = clocksource_delta(csnow, cs->cs_last, cs->mask);
  cs_nsec = clocksource_cyc2ns(delta, cs->mult, cs->shift);
  wdlast = cs->wd_last;
  cslast = cs->cs_last;
  cs->cs_last = csnow;
  cs->wd_last = wdnow;

  if (atomic_read(&watchdog_reset_pending))
   continue;


  if (abs(cs_nsec - wd_nsec) > WATCHDOG_THRESHOLD) {
   pr_warn("timekeeping watchdog on CPU%d: Marking clocksource '%s' as unstable because the skew is too large:\n",
    smp_processor_id(), cs->name);
   pr_warn("                      '%s' wd_now: %llx wd_last: %llx mask: %llx\n",
    watchdog->name, wdnow, wdlast, watchdog->mask);
   pr_warn("                      '%s' cs_now: %llx cs_last: %llx mask: %llx\n",
    cs->name, csnow, cslast, cs->mask);
   __clocksource_unstable(cs);
   continue;
  }

  if (!(cs->flags & CLOCK_SOURCE_VALID_FOR_HRES) &&
      (cs->flags & CLOCK_SOURCE_IS_CONTINUOUS) &&
      (watchdog->flags & CLOCK_SOURCE_IS_CONTINUOUS)) {

   cs->flags |= CLOCK_SOURCE_VALID_FOR_HRES;





   if (!finished_booting)
    continue;
   if (cs != curr_clocksource) {
    cs->flags |= CLOCK_SOURCE_RESELECT;
    schedule_work(&watchdog_work);
   } else {
    tick_clock_notify();
   }
  }
 }





 if (reset_pending)
  atomic_dec(&watchdog_reset_pending);





 next_cpu = cpumask_next(raw_smp_processor_id(), cpu_online_mask);
 if (next_cpu >= nr_cpu_ids)
  next_cpu = cpumask_first(cpu_online_mask);
 watchdog_timer.expires += WATCHDOG_INTERVAL;
 add_timer_on(&watchdog_timer, next_cpu);
out:
 spin_unlock(&watchdog_lock);
}

static inline void clocksource_start_watchdog(void)
{
 if (watchdog_running || !watchdog || list_empty(&watchdog_list))
  return;
 init_timer(&watchdog_timer);
 watchdog_timer.function = clocksource_watchdog;
 watchdog_timer.expires = jiffies + WATCHDOG_INTERVAL;
 add_timer_on(&watchdog_timer, cpumask_first(cpu_online_mask));
 watchdog_running = 1;
}

static inline void clocksource_stop_watchdog(void)
{
 if (!watchdog_running || (watchdog && !list_empty(&watchdog_list)))
  return;
 del_timer(&watchdog_timer);
 watchdog_running = 0;
}

static inline void clocksource_reset_watchdog(void)
{
 struct clocksource *cs;

 list_for_each_entry(cs, &watchdog_list, wd_list)
  cs->flags &= ~CLOCK_SOURCE_WATCHDOG;
}

static void clocksource_resume_watchdog(void)
{
 atomic_inc(&watchdog_reset_pending);
}

static void clocksource_enqueue_watchdog(struct clocksource *cs)
{
 unsigned long flags;

 spin_lock_irqsave(&watchdog_lock, flags);
 if (cs->flags & CLOCK_SOURCE_MUST_VERIFY) {

  list_add(&cs->wd_list, &watchdog_list);
  cs->flags &= ~CLOCK_SOURCE_WATCHDOG;
 } else {

  if (cs->flags & CLOCK_SOURCE_IS_CONTINUOUS)
   cs->flags |= CLOCK_SOURCE_VALID_FOR_HRES;
 }
 spin_unlock_irqrestore(&watchdog_lock, flags);
}

static void clocksource_select_watchdog(bool fallback)
{
 struct clocksource *cs, *old_wd;
 unsigned long flags;

 spin_lock_irqsave(&watchdog_lock, flags);

 old_wd = watchdog;
 if (fallback)
  watchdog = NULL;

 list_for_each_entry(cs, &clocksource_list, list) {

  if (cs->flags & CLOCK_SOURCE_MUST_VERIFY)
   continue;


  if (fallback && cs == old_wd)
   continue;


  if (!watchdog || cs->rating > watchdog->rating)
   watchdog = cs;
 }

 if (!watchdog)
  watchdog = old_wd;


 if (watchdog != old_wd)
  clocksource_reset_watchdog();


 clocksource_start_watchdog();
 spin_unlock_irqrestore(&watchdog_lock, flags);
}

static void clocksource_dequeue_watchdog(struct clocksource *cs)
{
 unsigned long flags;

 spin_lock_irqsave(&watchdog_lock, flags);
 if (cs != watchdog) {
  if (cs->flags & CLOCK_SOURCE_MUST_VERIFY) {

   list_del_init(&cs->wd_list);

   clocksource_stop_watchdog();
  }
 }
 spin_unlock_irqrestore(&watchdog_lock, flags);
}

static int __clocksource_watchdog_kthread(void)
{
 struct clocksource *cs, *tmp;
 unsigned long flags;
 LIST_HEAD(unstable);
 int select = 0;

 spin_lock_irqsave(&watchdog_lock, flags);
 list_for_each_entry_safe(cs, tmp, &watchdog_list, wd_list) {
  if (cs->flags & CLOCK_SOURCE_UNSTABLE) {
   list_del_init(&cs->wd_list);
   list_add(&cs->wd_list, &unstable);
   select = 1;
  }
  if (cs->flags & CLOCK_SOURCE_RESELECT) {
   cs->flags &= ~CLOCK_SOURCE_RESELECT;
   select = 1;
  }
 }

 clocksource_stop_watchdog();
 spin_unlock_irqrestore(&watchdog_lock, flags);


 list_for_each_entry_safe(cs, tmp, &unstable, wd_list) {
  list_del_init(&cs->wd_list);
  __clocksource_change_rating(cs, 0);
 }
 return select;
}

static int clocksource_watchdog_kthread(void *data)
{
 mutex_lock(&clocksource_mutex);
 if (__clocksource_watchdog_kthread())
  clocksource_select();
 mutex_unlock(&clocksource_mutex);
 return 0;
}

static bool clocksource_is_watchdog(struct clocksource *cs)
{
 return cs == watchdog;
}


static void clocksource_enqueue_watchdog(struct clocksource *cs)
{
 if (cs->flags & CLOCK_SOURCE_IS_CONTINUOUS)
  cs->flags |= CLOCK_SOURCE_VALID_FOR_HRES;
}

static void clocksource_select_watchdog(bool fallback) { }
static inline void clocksource_dequeue_watchdog(struct clocksource *cs) { }
static inline void clocksource_resume_watchdog(void) { }
static inline int __clocksource_watchdog_kthread(void) { return 0; }
static bool clocksource_is_watchdog(struct clocksource *cs) { return false; }
void clocksource_mark_unstable(struct clocksource *cs) { }





void clocksource_suspend(void)
{
 struct clocksource *cs;

 list_for_each_entry_reverse(cs, &clocksource_list, list)
  if (cs->suspend)
   cs->suspend(cs);
}




void clocksource_resume(void)
{
 struct clocksource *cs;

 list_for_each_entry(cs, &clocksource_list, list)
  if (cs->resume)
   cs->resume(cs);

 clocksource_resume_watchdog();
}
void clocksource_touch_watchdog(void)
{
 clocksource_resume_watchdog();
}






static u32 clocksource_max_adjustment(struct clocksource *cs)
{
 u64 ret;



 ret = (u64)cs->mult * 11;
 do_div(ret,100);
 return (u32)ret;
}
u64 clocks_calc_max_nsecs(u32 mult, u32 shift, u32 maxadj, u64 mask, u64 *max_cyc)
{
 u64 max_nsecs, max_cycles;





 max_cycles = ULLONG_MAX;
 do_div(max_cycles, mult+maxadj);







 max_cycles = min(max_cycles, mask);
 max_nsecs = clocksource_cyc2ns(max_cycles, mult - maxadj, shift);


 if (max_cyc)
  *max_cyc = max_cycles;


 max_nsecs >>= 1;

 return max_nsecs;
}






static inline void clocksource_update_max_deferment(struct clocksource *cs)
{
 cs->max_idle_ns = clocks_calc_max_nsecs(cs->mult, cs->shift,
      cs->maxadj, cs->mask,
      &cs->max_cycles);
}


static struct clocksource *clocksource_find_best(bool oneshot, bool skipcur)
{
 struct clocksource *cs;

 if (!finished_booting || list_empty(&clocksource_list))
  return NULL;






 list_for_each_entry(cs, &clocksource_list, list) {
  if (skipcur && cs == curr_clocksource)
   continue;
  if (oneshot && !(cs->flags & CLOCK_SOURCE_VALID_FOR_HRES))
   continue;
  return cs;
 }
 return NULL;
}

static void __clocksource_select(bool skipcur)
{
 bool oneshot = tick_oneshot_mode_active();
 struct clocksource *best, *cs;


 best = clocksource_find_best(oneshot, skipcur);
 if (!best)
  return;


 list_for_each_entry(cs, &clocksource_list, list) {
  if (skipcur && cs == curr_clocksource)
   continue;
  if (strcmp(cs->name, override_name) != 0)
   continue;





  if (!(cs->flags & CLOCK_SOURCE_VALID_FOR_HRES) && oneshot) {

   pr_warn("Override clocksource %s is not HRT compatible - cannot switch while in HRT/NOHZ mode\n",
    cs->name);
   override_name[0] = 0;
  } else

   best = cs;
  break;
 }

 if (curr_clocksource != best && !timekeeping_notify(best)) {
  pr_info("Switched to clocksource %s\n", best->name);
  curr_clocksource = best;
 }
}
static void clocksource_select(void)
{
 __clocksource_select(false);
}

static void clocksource_select_fallback(void)
{
 __clocksource_select(true);
}

static inline void clocksource_select(void) { }
static inline void clocksource_select_fallback(void) { }

static int __init clocksource_done_booting(void)
{
 mutex_lock(&clocksource_mutex);
 curr_clocksource = clocksource_default_clock();
 finished_booting = 1;



 __clocksource_watchdog_kthread();
 clocksource_select();
 mutex_unlock(&clocksource_mutex);
 return 0;
}
fs_initcall(clocksource_done_booting);




static void clocksource_enqueue(struct clocksource *cs)
{
 struct list_head *entry = &clocksource_list;
 struct clocksource *tmp;

 list_for_each_entry(tmp, &clocksource_list, list)

  if (tmp->rating >= cs->rating)
   entry = &tmp->list;
 list_add(&cs->list, entry);
}
void __clocksource_update_freq_scale(struct clocksource *cs, u32 scale, u32 freq)
{
 u64 sec;





 if (freq) {
  sec = cs->mask;
  do_div(sec, freq);
  do_div(sec, scale);
  if (!sec)
   sec = 1;
  else if (sec > 600 && cs->mask > UINT_MAX)
   sec = 600;

  clocks_calc_mult_shift(&cs->mult, &cs->shift, freq,
           NSEC_PER_SEC / scale, sec * scale);
 }




 cs->maxadj = clocksource_max_adjustment(cs);
 while (freq && ((cs->mult + cs->maxadj < cs->mult)
  || (cs->mult - cs->maxadj > cs->mult))) {
  cs->mult >>= 1;
  cs->shift--;
  cs->maxadj = clocksource_max_adjustment(cs);
 }





 WARN_ONCE(cs->mult + cs->maxadj < cs->mult,
  "timekeeping: Clocksource %s might overflow on 11%% adjustment\n",
  cs->name);

 clocksource_update_max_deferment(cs);

 pr_info("%s: mask: 0x%llx max_cycles: 0x%llx, max_idle_ns: %lld ns\n",
  cs->name, cs->mask, cs->max_cycles, cs->max_idle_ns);
}
EXPORT_SYMBOL_GPL(__clocksource_update_freq_scale);
int __clocksource_register_scale(struct clocksource *cs, u32 scale, u32 freq)
{


 __clocksource_update_freq_scale(cs, scale, freq);


 mutex_lock(&clocksource_mutex);
 clocksource_enqueue(cs);
 clocksource_enqueue_watchdog(cs);
 clocksource_select();
 clocksource_select_watchdog(false);
 mutex_unlock(&clocksource_mutex);
 return 0;
}
EXPORT_SYMBOL_GPL(__clocksource_register_scale);

static void __clocksource_change_rating(struct clocksource *cs, int rating)
{
 list_del(&cs->list);
 cs->rating = rating;
 clocksource_enqueue(cs);
}






void clocksource_change_rating(struct clocksource *cs, int rating)
{
 mutex_lock(&clocksource_mutex);
 __clocksource_change_rating(cs, rating);
 clocksource_select();
 clocksource_select_watchdog(false);
 mutex_unlock(&clocksource_mutex);
}
EXPORT_SYMBOL(clocksource_change_rating);




static int clocksource_unbind(struct clocksource *cs)
{
 if (clocksource_is_watchdog(cs)) {

  clocksource_select_watchdog(true);
  if (clocksource_is_watchdog(cs))
   return -EBUSY;
 }

 if (cs == curr_clocksource) {

  clocksource_select_fallback();
  if (curr_clocksource == cs)
   return -EBUSY;
 }
 clocksource_dequeue_watchdog(cs);
 list_del_init(&cs->list);
 return 0;
}





int clocksource_unregister(struct clocksource *cs)
{
 int ret = 0;

 mutex_lock(&clocksource_mutex);
 if (!list_empty(&cs->list))
  ret = clocksource_unbind(cs);
 mutex_unlock(&clocksource_mutex);
 return ret;
}
EXPORT_SYMBOL(clocksource_unregister);

static ssize_t
sysfs_show_current_clocksources(struct device *dev,
    struct device_attribute *attr, char *buf)
{
 ssize_t count = 0;

 mutex_lock(&clocksource_mutex);
 count = snprintf(buf, PAGE_SIZE, "%s\n", curr_clocksource->name);
 mutex_unlock(&clocksource_mutex);

 return count;
}

ssize_t sysfs_get_uname(const char *buf, char *dst, size_t cnt)
{
 size_t ret = cnt;


 if (!cnt || cnt >= CS_NAME_LEN)
  return -EINVAL;


 if (buf[cnt-1] == '\n')
  cnt--;
 if (cnt > 0)
  memcpy(dst, buf, cnt);
 dst[cnt] = 0;
 return ret;
}
static ssize_t sysfs_override_clocksource(struct device *dev,
       struct device_attribute *attr,
       const char *buf, size_t count)
{
 ssize_t ret;

 mutex_lock(&clocksource_mutex);

 ret = sysfs_get_uname(buf, override_name, count);
 if (ret >= 0)
  clocksource_select();

 mutex_unlock(&clocksource_mutex);

 return ret;
}
static ssize_t sysfs_unbind_clocksource(struct device *dev,
     struct device_attribute *attr,
     const char *buf, size_t count)
{
 struct clocksource *cs;
 char name[CS_NAME_LEN];
 ssize_t ret;

 ret = sysfs_get_uname(buf, name, count);
 if (ret < 0)
  return ret;

 ret = -ENODEV;
 mutex_lock(&clocksource_mutex);
 list_for_each_entry(cs, &clocksource_list, list) {
  if (strcmp(cs->name, name))
   continue;
  ret = clocksource_unbind(cs);
  break;
 }
 mutex_unlock(&clocksource_mutex);

 return ret ? ret : count;
}
static ssize_t
sysfs_show_available_clocksources(struct device *dev,
      struct device_attribute *attr,
      char *buf)
{
 struct clocksource *src;
 ssize_t count = 0;

 mutex_lock(&clocksource_mutex);
 list_for_each_entry(src, &clocksource_list, list) {




  if (!tick_oneshot_mode_active() ||
      (src->flags & CLOCK_SOURCE_VALID_FOR_HRES))
   count += snprintf(buf + count,
      max((ssize_t)PAGE_SIZE - count, (ssize_t)0),
      "%s ", src->name);
 }
 mutex_unlock(&clocksource_mutex);

 count += snprintf(buf + count,
     max((ssize_t)PAGE_SIZE - count, (ssize_t)0), "\n");

 return count;
}




static DEVICE_ATTR(current_clocksource, 0644, sysfs_show_current_clocksources,
     sysfs_override_clocksource);

static DEVICE_ATTR(unbind_clocksource, 0200, NULL, sysfs_unbind_clocksource);

static DEVICE_ATTR(available_clocksource, 0444,
     sysfs_show_available_clocksources, NULL);

static struct bus_type clocksource_subsys = {
 .name = "clocksource",
 .dev_name = "clocksource",
};

static struct device device_clocksource = {
 .id = 0,
 .bus = &clocksource_subsys,
};

static int __init init_clocksource_sysfs(void)
{
 int error = subsys_system_register(&clocksource_subsys, NULL);

 if (!error)
  error = device_register(&device_clocksource);
 if (!error)
  error = device_create_file(
    &device_clocksource,
    &dev_attr_current_clocksource);
 if (!error)
  error = device_create_file(&device_clocksource,
        &dev_attr_unbind_clocksource);
 if (!error)
  error = device_create_file(
    &device_clocksource,
    &dev_attr_available_clocksource);
 return error;
}

device_initcall(init_clocksource_sysfs);
static int __init boot_override_clocksource(char* str)
{
 mutex_lock(&clocksource_mutex);
 if (str)
  strlcpy(override_name, str, sizeof(override_name));
 mutex_unlock(&clocksource_mutex);
 return 1;
}

__setup("clocksource=", boot_override_clocksource);
static int __init boot_override_clock(char* str)
{
 if (!strcmp(str, "pmtmr")) {
  pr_warn("clock=pmtmr is deprecated - use clocksource=acpi_pm\n");
  return boot_override_clocksource("acpi_pm");
 }
 pr_warn("clock= boot option is deprecated - use clocksource=xyz\n");
 return boot_override_clocksource(str);
}

__setup("clock=", boot_override_clock);


static int compat_get_timex(struct timex *txc, struct compat_timex __user *utp)
{
 memset(txc, 0, sizeof(struct timex));

 if (!access_ok(VERIFY_READ, utp, sizeof(struct compat_timex)) ||
   __get_user(txc->modes, &utp->modes) ||
   __get_user(txc->offset, &utp->offset) ||
   __get_user(txc->freq, &utp->freq) ||
   __get_user(txc->maxerror, &utp->maxerror) ||
   __get_user(txc->esterror, &utp->esterror) ||
   __get_user(txc->status, &utp->status) ||
   __get_user(txc->constant, &utp->constant) ||
   __get_user(txc->precision, &utp->precision) ||
   __get_user(txc->tolerance, &utp->tolerance) ||
   __get_user(txc->time.tv_sec, &utp->time.tv_sec) ||
   __get_user(txc->time.tv_usec, &utp->time.tv_usec) ||
   __get_user(txc->tick, &utp->tick) ||
   __get_user(txc->ppsfreq, &utp->ppsfreq) ||
   __get_user(txc->jitter, &utp->jitter) ||
   __get_user(txc->shift, &utp->shift) ||
   __get_user(txc->stabil, &utp->stabil) ||
   __get_user(txc->jitcnt, &utp->jitcnt) ||
   __get_user(txc->calcnt, &utp->calcnt) ||
   __get_user(txc->errcnt, &utp->errcnt) ||
   __get_user(txc->stbcnt, &utp->stbcnt))
  return -EFAULT;

 return 0;
}

static int compat_put_timex(struct compat_timex __user *utp, struct timex *txc)
{
 if (!access_ok(VERIFY_WRITE, utp, sizeof(struct compat_timex)) ||
   __put_user(txc->modes, &utp->modes) ||
   __put_user(txc->offset, &utp->offset) ||
   __put_user(txc->freq, &utp->freq) ||
   __put_user(txc->maxerror, &utp->maxerror) ||
   __put_user(txc->esterror, &utp->esterror) ||
   __put_user(txc->status, &utp->status) ||
   __put_user(txc->constant, &utp->constant) ||
   __put_user(txc->precision, &utp->precision) ||
   __put_user(txc->tolerance, &utp->tolerance) ||
   __put_user(txc->time.tv_sec, &utp->time.tv_sec) ||
   __put_user(txc->time.tv_usec, &utp->time.tv_usec) ||
   __put_user(txc->tick, &utp->tick) ||
   __put_user(txc->ppsfreq, &utp->ppsfreq) ||
   __put_user(txc->jitter, &utp->jitter) ||
   __put_user(txc->shift, &utp->shift) ||
   __put_user(txc->stabil, &utp->stabil) ||
   __put_user(txc->jitcnt, &utp->jitcnt) ||
   __put_user(txc->calcnt, &utp->calcnt) ||
   __put_user(txc->errcnt, &utp->errcnt) ||
   __put_user(txc->stbcnt, &utp->stbcnt) ||
   __put_user(txc->tai, &utp->tai))
  return -EFAULT;
 return 0;
}

COMPAT_SYSCALL_DEFINE2(gettimeofday, struct compat_timeval __user *, tv,
         struct timezone __user *, tz)
{
 if (tv) {
  struct timeval ktv;
  do_gettimeofday(&ktv);
  if (compat_put_timeval(&ktv, tv))
   return -EFAULT;
 }
 if (tz) {
  if (copy_to_user(tz, &sys_tz, sizeof(sys_tz)))
   return -EFAULT;
 }

 return 0;
}

COMPAT_SYSCALL_DEFINE2(settimeofday, struct compat_timeval __user *, tv,
         struct timezone __user *, tz)
{
 struct timeval user_tv;
 struct timespec new_ts;
 struct timezone new_tz;

 if (tv) {
  if (compat_get_timeval(&user_tv, tv))
   return -EFAULT;
  new_ts.tv_sec = user_tv.tv_sec;
  new_ts.tv_nsec = user_tv.tv_usec * NSEC_PER_USEC;
 }
 if (tz) {
  if (copy_from_user(&new_tz, tz, sizeof(*tz)))
   return -EFAULT;
 }

 return do_sys_settimeofday(tv ? &new_ts : NULL, tz ? &new_tz : NULL);
}

static int __compat_get_timeval(struct timeval *tv, const struct compat_timeval __user *ctv)
{
 return (!access_ok(VERIFY_READ, ctv, sizeof(*ctv)) ||
   __get_user(tv->tv_sec, &ctv->tv_sec) ||
   __get_user(tv->tv_usec, &ctv->tv_usec)) ? -EFAULT : 0;
}

static int __compat_put_timeval(const struct timeval *tv, struct compat_timeval __user *ctv)
{
 return (!access_ok(VERIFY_WRITE, ctv, sizeof(*ctv)) ||
   __put_user(tv->tv_sec, &ctv->tv_sec) ||
   __put_user(tv->tv_usec, &ctv->tv_usec)) ? -EFAULT : 0;
}

static int __compat_get_timespec(struct timespec *ts, const struct compat_timespec __user *cts)
{
 return (!access_ok(VERIFY_READ, cts, sizeof(*cts)) ||
   __get_user(ts->tv_sec, &cts->tv_sec) ||
   __get_user(ts->tv_nsec, &cts->tv_nsec)) ? -EFAULT : 0;
}

static int __compat_put_timespec(const struct timespec *ts, struct compat_timespec __user *cts)
{
 return (!access_ok(VERIFY_WRITE, cts, sizeof(*cts)) ||
   __put_user(ts->tv_sec, &cts->tv_sec) ||
   __put_user(ts->tv_nsec, &cts->tv_nsec)) ? -EFAULT : 0;
}

int compat_get_timeval(struct timeval *tv, const void __user *utv)
{
 if (COMPAT_USE_64BIT_TIME)
  return copy_from_user(tv, utv, sizeof(*tv)) ? -EFAULT : 0;
 else
  return __compat_get_timeval(tv, utv);
}
EXPORT_SYMBOL_GPL(compat_get_timeval);

int compat_put_timeval(const struct timeval *tv, void __user *utv)
{
 if (COMPAT_USE_64BIT_TIME)
  return copy_to_user(utv, tv, sizeof(*tv)) ? -EFAULT : 0;
 else
  return __compat_put_timeval(tv, utv);
}
EXPORT_SYMBOL_GPL(compat_put_timeval);

int compat_get_timespec(struct timespec *ts, const void __user *uts)
{
 if (COMPAT_USE_64BIT_TIME)
  return copy_from_user(ts, uts, sizeof(*ts)) ? -EFAULT : 0;
 else
  return __compat_get_timespec(ts, uts);
}
EXPORT_SYMBOL_GPL(compat_get_timespec);

int compat_put_timespec(const struct timespec *ts, void __user *uts)
{
 if (COMPAT_USE_64BIT_TIME)
  return copy_to_user(uts, ts, sizeof(*ts)) ? -EFAULT : 0;
 else
  return __compat_put_timespec(ts, uts);
}
EXPORT_SYMBOL_GPL(compat_put_timespec);

int compat_convert_timespec(struct timespec __user **kts,
       const void __user *cts)
{
 struct timespec ts;
 struct timespec __user *uts;

 if (!cts || COMPAT_USE_64BIT_TIME) {
  *kts = (struct timespec __user *)cts;
  return 0;
 }

 uts = compat_alloc_user_space(sizeof(ts));
 if (!uts)
  return -EFAULT;
 if (compat_get_timespec(&ts, cts))
  return -EFAULT;
 if (copy_to_user(uts, &ts, sizeof(ts)))
  return -EFAULT;

 *kts = uts;
 return 0;
}

static long compat_nanosleep_restart(struct restart_block *restart)
{
 struct compat_timespec __user *rmtp;
 struct timespec rmt;
 mm_segment_t oldfs;
 long ret;

 restart->nanosleep.rmtp = (struct timespec __user *) &rmt;
 oldfs = get_fs();
 set_fs(KERNEL_DS);
 ret = hrtimer_nanosleep_restart(restart);
 set_fs(oldfs);

 if (ret == -ERESTART_RESTARTBLOCK) {
  rmtp = restart->nanosleep.compat_rmtp;

  if (rmtp && compat_put_timespec(&rmt, rmtp))
   return -EFAULT;
 }

 return ret;
}

COMPAT_SYSCALL_DEFINE2(nanosleep, struct compat_timespec __user *, rqtp,
         struct compat_timespec __user *, rmtp)
{
 struct timespec tu, rmt;
 mm_segment_t oldfs;
 long ret;

 if (compat_get_timespec(&tu, rqtp))
  return -EFAULT;

 if (!timespec_valid(&tu))
  return -EINVAL;

 oldfs = get_fs();
 set_fs(KERNEL_DS);
 ret = hrtimer_nanosleep(&tu,
    rmtp ? (struct timespec __user *)&rmt : NULL,
    HRTIMER_MODE_REL, CLOCK_MONOTONIC);
 set_fs(oldfs);
 if (ret == -ERESTART_RESTARTBLOCK) {
  struct restart_block *restart = &current->restart_block;

  restart->fn = compat_nanosleep_restart;
  restart->nanosleep.compat_rmtp = rmtp;

  if (rmtp && compat_put_timespec(&rmt, rmtp))
   return -EFAULT;
 }
 return ret;
}

static inline long get_compat_itimerval(struct itimerval *o,
  struct compat_itimerval __user *i)
{
 return (!access_ok(VERIFY_READ, i, sizeof(*i)) ||
  (__get_user(o->it_interval.tv_sec, &i->it_interval.tv_sec) |
   __get_user(o->it_interval.tv_usec, &i->it_interval.tv_usec) |
   __get_user(o->it_value.tv_sec, &i->it_value.tv_sec) |
   __get_user(o->it_value.tv_usec, &i->it_value.tv_usec)));
}

static inline long put_compat_itimerval(struct compat_itimerval __user *o,
  struct itimerval *i)
{
 return (!access_ok(VERIFY_WRITE, o, sizeof(*o)) ||
  (__put_user(i->it_interval.tv_sec, &o->it_interval.tv_sec) |
   __put_user(i->it_interval.tv_usec, &o->it_interval.tv_usec) |
   __put_user(i->it_value.tv_sec, &o->it_value.tv_sec) |
   __put_user(i->it_value.tv_usec, &o->it_value.tv_usec)));
}

COMPAT_SYSCALL_DEFINE2(getitimer, int, which,
  struct compat_itimerval __user *, it)
{
 struct itimerval kit;
 int error;

 error = do_getitimer(which, &kit);
 if (!error && put_compat_itimerval(it, &kit))
  error = -EFAULT;
 return error;
}

COMPAT_SYSCALL_DEFINE3(setitimer, int, which,
  struct compat_itimerval __user *, in,
  struct compat_itimerval __user *, out)
{
 struct itimerval kin, kout;
 int error;

 if (in) {
  if (get_compat_itimerval(&kin, in))
   return -EFAULT;
 } else
  memset(&kin, 0, sizeof(kin));

 error = do_setitimer(which, &kin, out ? &kout : NULL);
 if (error || !out)
  return error;
 if (put_compat_itimerval(out, &kout))
  return -EFAULT;
 return 0;
}

static compat_clock_t clock_t_to_compat_clock_t(clock_t x)
{
 return compat_jiffies_to_clock_t(clock_t_to_jiffies(x));
}

COMPAT_SYSCALL_DEFINE1(times, struct compat_tms __user *, tbuf)
{
 if (tbuf) {
  struct tms tms;
  struct compat_tms tmp;

  do_sys_times(&tms);

  tmp.tms_utime = clock_t_to_compat_clock_t(tms.tms_utime);
  tmp.tms_stime = clock_t_to_compat_clock_t(tms.tms_stime);
  tmp.tms_cutime = clock_t_to_compat_clock_t(tms.tms_cutime);
  tmp.tms_cstime = clock_t_to_compat_clock_t(tms.tms_cstime);
  if (copy_to_user(tbuf, &tmp, sizeof(tmp)))
   return -EFAULT;
 }
 force_successful_syscall_return();
 return compat_jiffies_to_clock_t(jiffies);
}







COMPAT_SYSCALL_DEFINE1(sigpending, compat_old_sigset_t __user *, set)
{
 old_sigset_t s;
 long ret;
 mm_segment_t old_fs = get_fs();

 set_fs(KERNEL_DS);
 ret = sys_sigpending((old_sigset_t __user *) &s);
 set_fs(old_fs);
 if (ret == 0)
  ret = put_user(s, set);
 return ret;
}







static inline void compat_sig_setmask(sigset_t *blocked, compat_sigset_word set)
{
 memcpy(blocked->sig, &set, sizeof(set));
}

COMPAT_SYSCALL_DEFINE3(sigprocmask, int, how,
         compat_old_sigset_t __user *, nset,
         compat_old_sigset_t __user *, oset)
{
 old_sigset_t old_set, new_set;
 sigset_t new_blocked;

 old_set = current->blocked.sig[0];

 if (nset) {
  if (get_user(new_set, nset))
   return -EFAULT;
  new_set &= ~(sigmask(SIGKILL) | sigmask(SIGSTOP));

  new_blocked = current->blocked;

  switch (how) {
  case SIG_BLOCK:
   sigaddsetmask(&new_blocked, new_set);
   break;
  case SIG_UNBLOCK:
   sigdelsetmask(&new_blocked, new_set);
   break;
  case SIG_SETMASK:
   compat_sig_setmask(&new_blocked, new_set);
   break;
  default:
   return -EINVAL;
  }

  set_current_blocked(&new_blocked);
 }

 if (oset) {
  if (put_user(old_set, oset))
   return -EFAULT;
 }

 return 0;
}


COMPAT_SYSCALL_DEFINE2(setrlimit, unsigned int, resource,
         struct compat_rlimit __user *, rlim)
{
 struct rlimit r;

 if (!access_ok(VERIFY_READ, rlim, sizeof(*rlim)) ||
     __get_user(r.rlim_cur, &rlim->rlim_cur) ||
     __get_user(r.rlim_max, &rlim->rlim_max))
  return -EFAULT;

 if (r.rlim_cur == COMPAT_RLIM_INFINITY)
  r.rlim_cur = RLIM_INFINITY;
 if (r.rlim_max == COMPAT_RLIM_INFINITY)
  r.rlim_max = RLIM_INFINITY;
 return do_prlimit(current, resource, &r, NULL);
}


COMPAT_SYSCALL_DEFINE2(old_getrlimit, unsigned int, resource,
         struct compat_rlimit __user *, rlim)
{
 struct rlimit r;
 int ret;
 mm_segment_t old_fs = get_fs();

 set_fs(KERNEL_DS);
 ret = sys_old_getrlimit(resource, (struct rlimit __user *)&r);
 set_fs(old_fs);

 if (!ret) {
  if (r.rlim_cur > COMPAT_RLIM_OLD_INFINITY)
   r.rlim_cur = COMPAT_RLIM_INFINITY;
  if (r.rlim_max > COMPAT_RLIM_OLD_INFINITY)
   r.rlim_max = COMPAT_RLIM_INFINITY;

  if (!access_ok(VERIFY_WRITE, rlim, sizeof(*rlim)) ||
      __put_user(r.rlim_cur, &rlim->rlim_cur) ||
      __put_user(r.rlim_max, &rlim->rlim_max))
   return -EFAULT;
 }
 return ret;
}


COMPAT_SYSCALL_DEFINE2(getrlimit, unsigned int, resource,
         struct compat_rlimit __user *, rlim)
{
 struct rlimit r;
 int ret;

 ret = do_prlimit(current, resource, NULL, &r);
 if (!ret) {
  if (r.rlim_cur > COMPAT_RLIM_INFINITY)
   r.rlim_cur = COMPAT_RLIM_INFINITY;
  if (r.rlim_max > COMPAT_RLIM_INFINITY)
   r.rlim_max = COMPAT_RLIM_INFINITY;

  if (!access_ok(VERIFY_WRITE, rlim, sizeof(*rlim)) ||
      __put_user(r.rlim_cur, &rlim->rlim_cur) ||
      __put_user(r.rlim_max, &rlim->rlim_max))
   return -EFAULT;
 }
 return ret;
}

int put_compat_rusage(const struct rusage *r, struct compat_rusage __user *ru)
{
 if (!access_ok(VERIFY_WRITE, ru, sizeof(*ru)) ||
     __put_user(r->ru_utime.tv_sec, &ru->ru_utime.tv_sec) ||
     __put_user(r->ru_utime.tv_usec, &ru->ru_utime.tv_usec) ||
     __put_user(r->ru_stime.tv_sec, &ru->ru_stime.tv_sec) ||
     __put_user(r->ru_stime.tv_usec, &ru->ru_stime.tv_usec) ||
     __put_user(r->ru_maxrss, &ru->ru_maxrss) ||
     __put_user(r->ru_ixrss, &ru->ru_ixrss) ||
     __put_user(r->ru_idrss, &ru->ru_idrss) ||
     __put_user(r->ru_isrss, &ru->ru_isrss) ||
     __put_user(r->ru_minflt, &ru->ru_minflt) ||
     __put_user(r->ru_majflt, &ru->ru_majflt) ||
     __put_user(r->ru_nswap, &ru->ru_nswap) ||
     __put_user(r->ru_inblock, &ru->ru_inblock) ||
     __put_user(r->ru_oublock, &ru->ru_oublock) ||
     __put_user(r->ru_msgsnd, &ru->ru_msgsnd) ||
     __put_user(r->ru_msgrcv, &ru->ru_msgrcv) ||
     __put_user(r->ru_nsignals, &ru->ru_nsignals) ||
     __put_user(r->ru_nvcsw, &ru->ru_nvcsw) ||
     __put_user(r->ru_nivcsw, &ru->ru_nivcsw))
  return -EFAULT;
 return 0;
}

COMPAT_SYSCALL_DEFINE4(wait4,
 compat_pid_t, pid,
 compat_uint_t __user *, stat_addr,
 int, options,
 struct compat_rusage __user *, ru)
{
 if (!ru) {
  return sys_wait4(pid, stat_addr, options, NULL);
 } else {
  struct rusage r;
  int ret;
  unsigned int status;
  mm_segment_t old_fs = get_fs();

  set_fs (KERNEL_DS);
  ret = sys_wait4(pid,
    (stat_addr ?
     (unsigned int __user *) &status : NULL),
    options, (struct rusage __user *) &r);
  set_fs (old_fs);

  if (ret > 0) {
   if (put_compat_rusage(&r, ru))
    return -EFAULT;
   if (stat_addr && put_user(status, stat_addr))
    return -EFAULT;
  }
  return ret;
 }
}

COMPAT_SYSCALL_DEFINE5(waitid,
  int, which, compat_pid_t, pid,
  struct compat_siginfo __user *, uinfo, int, options,
  struct compat_rusage __user *, uru)
{
 siginfo_t info;
 struct rusage ru;
 long ret;
 mm_segment_t old_fs = get_fs();

 memset(&info, 0, sizeof(info));

 set_fs(KERNEL_DS);
 ret = sys_waitid(which, pid, (siginfo_t __user *)&info, options,
    uru ? (struct rusage __user *)&ru : NULL);
 set_fs(old_fs);

 if ((ret < 0) || (info.si_signo == 0))
  return ret;

 if (uru) {

  if (COMPAT_USE_64BIT_TIME)
   ret = copy_to_user(uru, &ru, sizeof(ru));
  else
   ret = put_compat_rusage(&ru, uru);
  if (ret)
   return -EFAULT;
 }

 BUG_ON(info.si_code & __SI_MASK);
 info.si_code |= __SI_CHLD;
 return copy_siginfo_to_user32(uinfo, &info);
}

static int compat_get_user_cpu_mask(compat_ulong_t __user *user_mask_ptr,
        unsigned len, struct cpumask *new_mask)
{
 unsigned long *k;

 if (len < cpumask_size())
  memset(new_mask, 0, cpumask_size());
 else if (len > cpumask_size())
  len = cpumask_size();

 k = cpumask_bits(new_mask);
 return compat_get_bitmap(k, user_mask_ptr, len * 8);
}

COMPAT_SYSCALL_DEFINE3(sched_setaffinity, compat_pid_t, pid,
         unsigned int, len,
         compat_ulong_t __user *, user_mask_ptr)
{
 cpumask_var_t new_mask;
 int retval;

 if (!alloc_cpumask_var(&new_mask, GFP_KERNEL))
  return -ENOMEM;

 retval = compat_get_user_cpu_mask(user_mask_ptr, len, new_mask);
 if (retval)
  goto out;

 retval = sched_setaffinity(pid, new_mask);
out:
 free_cpumask_var(new_mask);
 return retval;
}

COMPAT_SYSCALL_DEFINE3(sched_getaffinity, compat_pid_t, pid, unsigned int, len,
         compat_ulong_t __user *, user_mask_ptr)
{
 int ret;
 cpumask_var_t mask;

 if ((len * BITS_PER_BYTE) < nr_cpu_ids)
  return -EINVAL;
 if (len & (sizeof(compat_ulong_t)-1))
  return -EINVAL;

 if (!alloc_cpumask_var(&mask, GFP_KERNEL))
  return -ENOMEM;

 ret = sched_getaffinity(pid, mask);
 if (ret == 0) {
  size_t retlen = min_t(size_t, len, cpumask_size());

  if (compat_put_bitmap(user_mask_ptr, cpumask_bits(mask), retlen * 8))
   ret = -EFAULT;
  else
   ret = retlen;
 }
 free_cpumask_var(mask);

 return ret;
}

int get_compat_itimerspec(struct itimerspec *dst,
     const struct compat_itimerspec __user *src)
{
 if (__compat_get_timespec(&dst->it_interval, &src->it_interval) ||
     __compat_get_timespec(&dst->it_value, &src->it_value))
  return -EFAULT;
 return 0;
}

int put_compat_itimerspec(struct compat_itimerspec __user *dst,
     const struct itimerspec *src)
{
 if (__compat_put_timespec(&src->it_interval, &dst->it_interval) ||
     __compat_put_timespec(&src->it_value, &dst->it_value))
  return -EFAULT;
 return 0;
}

COMPAT_SYSCALL_DEFINE3(timer_create, clockid_t, which_clock,
         struct compat_sigevent __user *, timer_event_spec,
         timer_t __user *, created_timer_id)
{
 struct sigevent __user *event = NULL;

 if (timer_event_spec) {
  struct sigevent kevent;

  event = compat_alloc_user_space(sizeof(*event));
  if (get_compat_sigevent(&kevent, timer_event_spec) ||
      copy_to_user(event, &kevent, sizeof(*event)))
   return -EFAULT;
 }

 return sys_timer_create(which_clock, event, created_timer_id);
}

COMPAT_SYSCALL_DEFINE4(timer_settime, timer_t, timer_id, int, flags,
         struct compat_itimerspec __user *, new,
         struct compat_itimerspec __user *, old)
{
 long err;
 mm_segment_t oldfs;
 struct itimerspec newts, oldts;

 if (!new)
  return -EINVAL;
 if (get_compat_itimerspec(&newts, new))
  return -EFAULT;
 oldfs = get_fs();
 set_fs(KERNEL_DS);
 err = sys_timer_settime(timer_id, flags,
    (struct itimerspec __user *) &newts,
    (struct itimerspec __user *) &oldts);
 set_fs(oldfs);
 if (!err && old && put_compat_itimerspec(old, &oldts))
  return -EFAULT;
 return err;
}

COMPAT_SYSCALL_DEFINE2(timer_gettime, timer_t, timer_id,
         struct compat_itimerspec __user *, setting)
{
 long err;
 mm_segment_t oldfs;
 struct itimerspec ts;

 oldfs = get_fs();
 set_fs(KERNEL_DS);
 err = sys_timer_gettime(timer_id,
    (struct itimerspec __user *) &ts);
 set_fs(oldfs);
 if (!err && put_compat_itimerspec(setting, &ts))
  return -EFAULT;
 return err;
}

COMPAT_SYSCALL_DEFINE2(clock_settime, clockid_t, which_clock,
         struct compat_timespec __user *, tp)
{
 long err;
 mm_segment_t oldfs;
 struct timespec ts;

 if (compat_get_timespec(&ts, tp))
  return -EFAULT;
 oldfs = get_fs();
 set_fs(KERNEL_DS);
 err = sys_clock_settime(which_clock,
    (struct timespec __user *) &ts);
 set_fs(oldfs);
 return err;
}

COMPAT_SYSCALL_DEFINE2(clock_gettime, clockid_t, which_clock,
         struct compat_timespec __user *, tp)
{
 long err;
 mm_segment_t oldfs;
 struct timespec ts;

 oldfs = get_fs();
 set_fs(KERNEL_DS);
 err = sys_clock_gettime(which_clock,
    (struct timespec __user *) &ts);
 set_fs(oldfs);
 if (!err && compat_put_timespec(&ts, tp))
  return -EFAULT;
 return err;
}

COMPAT_SYSCALL_DEFINE2(clock_adjtime, clockid_t, which_clock,
         struct compat_timex __user *, utp)
{
 struct timex txc;
 mm_segment_t oldfs;
 int err, ret;

 err = compat_get_timex(&txc, utp);
 if (err)
  return err;

 oldfs = get_fs();
 set_fs(KERNEL_DS);
 ret = sys_clock_adjtime(which_clock, (struct timex __user *) &txc);
 set_fs(oldfs);

 err = compat_put_timex(utp, &txc);
 if (err)
  return err;

 return ret;
}

COMPAT_SYSCALL_DEFINE2(clock_getres, clockid_t, which_clock,
         struct compat_timespec __user *, tp)
{
 long err;
 mm_segment_t oldfs;
 struct timespec ts;

 oldfs = get_fs();
 set_fs(KERNEL_DS);
 err = sys_clock_getres(which_clock,
          (struct timespec __user *) &ts);
 set_fs(oldfs);
 if (!err && tp && compat_put_timespec(&ts, tp))
  return -EFAULT;
 return err;
}

static long compat_clock_nanosleep_restart(struct restart_block *restart)
{
 long err;
 mm_segment_t oldfs;
 struct timespec tu;
 struct compat_timespec __user *rmtp = restart->nanosleep.compat_rmtp;

 restart->nanosleep.rmtp = (struct timespec __user *) &tu;
 oldfs = get_fs();
 set_fs(KERNEL_DS);
 err = clock_nanosleep_restart(restart);
 set_fs(oldfs);

 if ((err == -ERESTART_RESTARTBLOCK) && rmtp &&
     compat_put_timespec(&tu, rmtp))
  return -EFAULT;

 if (err == -ERESTART_RESTARTBLOCK) {
  restart->fn = compat_clock_nanosleep_restart;
  restart->nanosleep.compat_rmtp = rmtp;
 }
 return err;
}

COMPAT_SYSCALL_DEFINE4(clock_nanosleep, clockid_t, which_clock, int, flags,
         struct compat_timespec __user *, rqtp,
         struct compat_timespec __user *, rmtp)
{
 long err;
 mm_segment_t oldfs;
 struct timespec in, out;
 struct restart_block *restart;

 if (compat_get_timespec(&in, rqtp))
  return -EFAULT;

 oldfs = get_fs();
 set_fs(KERNEL_DS);
 err = sys_clock_nanosleep(which_clock, flags,
      (struct timespec __user *) &in,
      (struct timespec __user *) &out);
 set_fs(oldfs);

 if ((err == -ERESTART_RESTARTBLOCK) && rmtp &&
     compat_put_timespec(&out, rmtp))
  return -EFAULT;

 if (err == -ERESTART_RESTARTBLOCK) {
  restart = &current->restart_block;
  restart->fn = compat_clock_nanosleep_restart;
  restart->nanosleep.compat_rmtp = rmtp;
 }
 return err;
}
int get_compat_sigevent(struct sigevent *event,
  const struct compat_sigevent __user *u_event)
{
 memset(event, 0, sizeof(*event));
 return (!access_ok(VERIFY_READ, u_event, sizeof(*u_event)) ||
  __get_user(event->sigev_value.sival_int,
   &u_event->sigev_value.sival_int) ||
  __get_user(event->sigev_signo, &u_event->sigev_signo) ||
  __get_user(event->sigev_notify, &u_event->sigev_notify) ||
  __get_user(event->sigev_notify_thread_id,
   &u_event->sigev_notify_thread_id))
  ? -EFAULT : 0;
}

long compat_get_bitmap(unsigned long *mask, const compat_ulong_t __user *umask,
         unsigned long bitmap_size)
{
 int i, j;
 unsigned long m;
 compat_ulong_t um;
 unsigned long nr_compat_longs;


 bitmap_size = ALIGN(bitmap_size, BITS_PER_COMPAT_LONG);

 if (!access_ok(VERIFY_READ, umask, bitmap_size / 8))
  return -EFAULT;

 nr_compat_longs = BITS_TO_COMPAT_LONGS(bitmap_size);

 for (i = 0; i < BITS_TO_LONGS(bitmap_size); i++) {
  m = 0;

  for (j = 0; j < sizeof(m)/sizeof(um); j++) {





   if (nr_compat_longs) {
    nr_compat_longs--;
    if (__get_user(um, umask))
     return -EFAULT;
   } else {
    um = 0;
   }

   umask++;
   m |= (long)um << (j * BITS_PER_COMPAT_LONG);
  }
  *mask++ = m;
 }

 return 0;
}

long compat_put_bitmap(compat_ulong_t __user *umask, unsigned long *mask,
         unsigned long bitmap_size)
{
 int i, j;
 unsigned long m;
 compat_ulong_t um;
 unsigned long nr_compat_longs;


 bitmap_size = ALIGN(bitmap_size, BITS_PER_COMPAT_LONG);

 if (!access_ok(VERIFY_WRITE, umask, bitmap_size / 8))
  return -EFAULT;

 nr_compat_longs = BITS_TO_COMPAT_LONGS(bitmap_size);

 for (i = 0; i < BITS_TO_LONGS(bitmap_size); i++) {
  m = *mask++;

  for (j = 0; j < sizeof(m)/sizeof(um); j++) {
   um = m;





   if (nr_compat_longs) {
    nr_compat_longs--;
    if (__put_user(um, umask))
     return -EFAULT;
   }

   umask++;
   m >>= 4*sizeof(um);
   m >>= 4*sizeof(um);
  }
 }

 return 0;
}

void
sigset_from_compat(sigset_t *set, const compat_sigset_t *compat)
{
 switch (_NSIG_WORDS) {
 case 4: set->sig[3] = compat->sig[6] | (((long)compat->sig[7]) << 32 );
 case 3: set->sig[2] = compat->sig[4] | (((long)compat->sig[5]) << 32 );
 case 2: set->sig[1] = compat->sig[2] | (((long)compat->sig[3]) << 32 );
 case 1: set->sig[0] = compat->sig[0] | (((long)compat->sig[1]) << 32 );
 }
}
EXPORT_SYMBOL_GPL(sigset_from_compat);

void
sigset_to_compat(compat_sigset_t *compat, const sigset_t *set)
{
 switch (_NSIG_WORDS) {
 case 4: compat->sig[7] = (set->sig[3] >> 32); compat->sig[6] = set->sig[3];
 case 3: compat->sig[5] = (set->sig[2] >> 32); compat->sig[4] = set->sig[2];
 case 2: compat->sig[3] = (set->sig[1] >> 32); compat->sig[2] = set->sig[1];
 case 1: compat->sig[1] = (set->sig[0] >> 32); compat->sig[0] = set->sig[0];
 }
}

COMPAT_SYSCALL_DEFINE4(rt_sigtimedwait, compat_sigset_t __user *, uthese,
  struct compat_siginfo __user *, uinfo,
  struct compat_timespec __user *, uts, compat_size_t, sigsetsize)
{
 compat_sigset_t s32;
 sigset_t s;
 struct timespec t;
 siginfo_t info;
 long ret;

 if (sigsetsize != sizeof(sigset_t))
  return -EINVAL;

 if (copy_from_user(&s32, uthese, sizeof(compat_sigset_t)))
  return -EFAULT;
 sigset_from_compat(&s, &s32);

 if (uts) {
  if (compat_get_timespec(&t, uts))
   return -EFAULT;
 }

 ret = do_sigtimedwait(&s, &info, uts ? &t : NULL);

 if (ret > 0 && uinfo) {
  if (copy_siginfo_to_user32(uinfo, &info))
   ret = -EFAULT;
 }

 return ret;
}




COMPAT_SYSCALL_DEFINE1(time, compat_time_t __user *, tloc)
{
 compat_time_t i;
 struct timeval tv;

 do_gettimeofday(&tv);
 i = tv.tv_sec;

 if (tloc) {
  if (put_user(i,tloc))
   return -EFAULT;
 }
 force_successful_syscall_return();
 return i;
}

COMPAT_SYSCALL_DEFINE1(stime, compat_time_t __user *, tptr)
{
 struct timespec tv;
 int err;

 if (get_user(tv.tv_sec, tptr))
  return -EFAULT;

 tv.tv_nsec = 0;

 err = security_settime(&tv, NULL);
 if (err)
  return err;

 do_settimeofday(&tv);
 return 0;
}


COMPAT_SYSCALL_DEFINE1(adjtimex, struct compat_timex __user *, utp)
{
 struct timex txc;
 int err, ret;

 err = compat_get_timex(&txc, utp);
 if (err)
  return err;

 ret = do_adjtimex(&txc);

 err = compat_put_timex(utp, &txc);
 if (err)
  return err;

 return ret;
}

COMPAT_SYSCALL_DEFINE6(move_pages, pid_t, pid, compat_ulong_t, nr_pages,
         compat_uptr_t __user *, pages32,
         const int __user *, nodes,
         int __user *, status,
         int, flags)
{
 const void __user * __user *pages;
 int i;

 pages = compat_alloc_user_space(nr_pages * sizeof(void *));
 for (i = 0; i < nr_pages; i++) {
  compat_uptr_t p;

  if (get_user(p, pages32 + i) ||
   put_user(compat_ptr(p), pages + i))
   return -EFAULT;
 }
 return sys_move_pages(pid, nr_pages, pages, nodes, status, flags);
}

COMPAT_SYSCALL_DEFINE4(migrate_pages, compat_pid_t, pid,
         compat_ulong_t, maxnode,
         const compat_ulong_t __user *, old_nodes,
         const compat_ulong_t __user *, new_nodes)
{
 unsigned long __user *old = NULL;
 unsigned long __user *new = NULL;
 nodemask_t tmp_mask;
 unsigned long nr_bits;
 unsigned long size;

 nr_bits = min_t(unsigned long, maxnode - 1, MAX_NUMNODES);
 size = ALIGN(nr_bits, BITS_PER_LONG) / 8;
 if (old_nodes) {
  if (compat_get_bitmap(nodes_addr(tmp_mask), old_nodes, nr_bits))
   return -EFAULT;
  old = compat_alloc_user_space(new_nodes ? size * 2 : size);
  if (new_nodes)
   new = old + size / sizeof(unsigned long);
  if (copy_to_user(old, nodes_addr(tmp_mask), size))
   return -EFAULT;
 }
 if (new_nodes) {
  if (compat_get_bitmap(nodes_addr(tmp_mask), new_nodes, nr_bits))
   return -EFAULT;
  if (new == NULL)
   new = compat_alloc_user_space(size);
  if (copy_to_user(new, nodes_addr(tmp_mask), size))
   return -EFAULT;
 }
 return sys_migrate_pages(pid, nr_bits + 1, old, new);
}

COMPAT_SYSCALL_DEFINE2(sched_rr_get_interval,
         compat_pid_t, pid,
         struct compat_timespec __user *, interval)
{
 struct timespec t;
 int ret;
 mm_segment_t old_fs = get_fs();

 set_fs(KERNEL_DS);
 ret = sys_sched_rr_get_interval(pid, (struct timespec __user *)&t);
 set_fs(old_fs);
 if (compat_put_timespec(&t, interval))
  return -EFAULT;
 return ret;
}





void __user *compat_alloc_user_space(unsigned long len)
{
 void __user *ptr;


 if (unlikely(len > (((compat_uptr_t)~0) >> 1)))
  return NULL;

 ptr = arch_compat_alloc_user_space(len);

 if (unlikely(!access_ok(VERIFY_WRITE, ptr, len)))
  return NULL;

 return ptr;
}
EXPORT_SYMBOL_GPL(compat_alloc_user_space);


 (sizeof(kernel_config_data) - 1 - MAGIC_SIZE * 2)


static ssize_t
ikconfig_read_current(struct file *file, char __user *buf,
        size_t len, loff_t * offset)
{
 return simple_read_from_buffer(buf, len, offset,
           kernel_config_data + MAGIC_SIZE,
           kernel_config_data_size);
}

static const struct file_operations ikconfig_file_ops = {
 .owner = THIS_MODULE,
 .read = ikconfig_read_current,
 .llseek = default_llseek,
};

static int __init ikconfig_init(void)
{
 struct proc_dir_entry *entry;


 entry = proc_create("config.gz", S_IFREG | S_IRUGO, NULL,
       &ikconfig_file_ops);
 if (!entry)
  return -ENOMEM;

 proc_set_size(entry, kernel_config_data_size);

 return 0;
}

static void __exit ikconfig_cleanup(void)
{
 remove_proc_entry("config.gz", NULL);
}

module_init(ikconfig_init);
module_exit(ikconfig_cleanup);


MODULE_LICENSE("GPL");
MODULE_AUTHOR("Randy Dunlap");
MODULE_DESCRIPTION("Echo the kernel .config file used to build the kernel");








static int orig_fgconsole, orig_kmsg;

static DEFINE_MUTEX(vt_switch_mutex);

struct pm_vt_switch {
 struct list_head head;
 struct device *dev;
 bool required;
};

static LIST_HEAD(pm_vt_switch_list);
void pm_vt_switch_required(struct device *dev, bool required)
{
 struct pm_vt_switch *entry, *tmp;

 mutex_lock(&vt_switch_mutex);
 list_for_each_entry(tmp, &pm_vt_switch_list, head) {
  if (tmp->dev == dev) {

   tmp->required = required;
   goto out;
  }
 }

 entry = kmalloc(sizeof(*entry), GFP_KERNEL);
 if (!entry)
  goto out;

 entry->required = required;
 entry->dev = dev;

 list_add(&entry->head, &pm_vt_switch_list);
out:
 mutex_unlock(&vt_switch_mutex);
}
EXPORT_SYMBOL(pm_vt_switch_required);







void pm_vt_switch_unregister(struct device *dev)
{
 struct pm_vt_switch *tmp;

 mutex_lock(&vt_switch_mutex);
 list_for_each_entry(tmp, &pm_vt_switch_list, head) {
  if (tmp->dev == dev) {
   list_del(&tmp->head);
   kfree(tmp);
   break;
  }
 }
 mutex_unlock(&vt_switch_mutex);
}
EXPORT_SYMBOL(pm_vt_switch_unregister);
static bool pm_vt_switch(void)
{
 struct pm_vt_switch *entry;
 bool ret = true;

 mutex_lock(&vt_switch_mutex);
 if (list_empty(&pm_vt_switch_list))
  goto out;

 if (!console_suspend_enabled)
  goto out;

 list_for_each_entry(entry, &pm_vt_switch_list, head) {
  if (entry->required)
   goto out;
 }

 ret = false;
out:
 mutex_unlock(&vt_switch_mutex);
 return ret;
}

int pm_prepare_console(void)
{
 if (!pm_vt_switch())
  return 0;

 orig_fgconsole = vt_move_to_console(SUSPEND_CONSOLE, 1);
 if (orig_fgconsole < 0)
  return 1;

 orig_kmsg = vt_kmsg_redirect(SUSPEND_CONSOLE);
 return 0;
}

void pm_restore_console(void)
{
 if (!pm_vt_switch())
  return;

 if (orig_fgconsole >= 0) {
  vt_move_to_console(orig_fgconsole, 0);
  vt_kmsg_redirect(orig_kmsg);
 }
}


DEFINE_STATIC_KEY_FALSE(context_tracking_enabled);
EXPORT_SYMBOL_GPL(context_tracking_enabled);

DEFINE_PER_CPU(struct context_tracking, context_tracking);
EXPORT_SYMBOL_GPL(context_tracking);

static bool context_tracking_recursion_enter(void)
{
 int recursion;

 recursion = __this_cpu_inc_return(context_tracking.recursion);
 if (recursion == 1)
  return true;

 WARN_ONCE((recursion < 1), "Invalid context tracking recursion value %d\n", recursion);
 __this_cpu_dec(context_tracking.recursion);

 return false;
}

static void context_tracking_recursion_exit(void)
{
 __this_cpu_dec(context_tracking.recursion);
}
void __context_tracking_enter(enum ctx_state state)
{

 WARN_ON_ONCE(!current->mm);

 if (!context_tracking_recursion_enter())
  return;

 if ( __this_cpu_read(context_tracking.state) != state) {
  if (__this_cpu_read(context_tracking.active)) {







   if (state == CONTEXT_USER) {
    trace_user_enter(0);
    vtime_user_enter(current);
   }
   rcu_user_enter();
  }
  __this_cpu_write(context_tracking.state, state);
 }
 context_tracking_recursion_exit();
}
NOKPROBE_SYMBOL(__context_tracking_enter);
EXPORT_SYMBOL_GPL(__context_tracking_enter);

void context_tracking_enter(enum ctx_state state)
{
 unsigned long flags;
 if (in_interrupt())
  return;

 local_irq_save(flags);
 __context_tracking_enter(state);
 local_irq_restore(flags);
}
NOKPROBE_SYMBOL(context_tracking_enter);
EXPORT_SYMBOL_GPL(context_tracking_enter);

void context_tracking_user_enter(void)
{
 user_enter();
}
NOKPROBE_SYMBOL(context_tracking_user_enter);
void __context_tracking_exit(enum ctx_state state)
{
 if (!context_tracking_recursion_enter())
  return;

 if (__this_cpu_read(context_tracking.state) == state) {
  if (__this_cpu_read(context_tracking.active)) {




   rcu_user_exit();
   if (state == CONTEXT_USER) {
    vtime_user_exit(current);
    trace_user_exit(0);
   }
  }
  __this_cpu_write(context_tracking.state, CONTEXT_KERNEL);
 }
 context_tracking_recursion_exit();
}
NOKPROBE_SYMBOL(__context_tracking_exit);
EXPORT_SYMBOL_GPL(__context_tracking_exit);

void context_tracking_exit(enum ctx_state state)
{
 unsigned long flags;

 if (in_interrupt())
  return;

 local_irq_save(flags);
 __context_tracking_exit(state);
 local_irq_restore(flags);
}
NOKPROBE_SYMBOL(context_tracking_exit);
EXPORT_SYMBOL_GPL(context_tracking_exit);

void context_tracking_user_exit(void)
{
 user_exit();
}
NOKPROBE_SYMBOL(context_tracking_user_exit);

void __init context_tracking_cpu_set(int cpu)
{
 static __initdata bool initialized = false;

 if (!per_cpu(context_tracking.active, cpu)) {
  per_cpu(context_tracking.active, cpu) = true;
  static_branch_inc(&context_tracking_enabled);
 }

 if (initialized)
  return;





 set_tsk_thread_flag(&init_task, TIF_NOHZ);
 WARN_ON_ONCE(!tasklist_empty());

 initialized = true;
}

void __init context_tracking_init(void)
{
 int cpu;

 for_each_possible_cpu(cpu)
  context_tracking_cpu_set(cpu);
}










void *bpf_internal_load_pointer_neg_helper(const struct sk_buff *skb, int k, unsigned int size)
{
 u8 *ptr = NULL;

 if (k >= SKF_NET_OFF)
  ptr = skb_network_header(skb) + k - SKF_NET_OFF;
 else if (k >= SKF_LL_OFF)
  ptr = skb_mac_header(skb) + k - SKF_LL_OFF;

 if (ptr >= skb->head && ptr + size <= skb_tail_pointer(skb))
  return ptr;

 return NULL;
}

struct bpf_prog *bpf_prog_alloc(unsigned int size, gfp_t gfp_extra_flags)
{
 gfp_t gfp_flags = GFP_KERNEL | __GFP_HIGHMEM | __GFP_ZERO |
     gfp_extra_flags;
 struct bpf_prog_aux *aux;
 struct bpf_prog *fp;

 size = round_up(size, PAGE_SIZE);
 fp = __vmalloc(size, gfp_flags, PAGE_KERNEL);
 if (fp == NULL)
  return NULL;

 kmemcheck_annotate_bitfield(fp, meta);

 aux = kzalloc(sizeof(*aux), GFP_KERNEL | gfp_extra_flags);
 if (aux == NULL) {
  vfree(fp);
  return NULL;
 }

 fp->pages = size / PAGE_SIZE;
 fp->aux = aux;
 fp->aux->prog = fp;

 return fp;
}
EXPORT_SYMBOL_GPL(bpf_prog_alloc);

struct bpf_prog *bpf_prog_realloc(struct bpf_prog *fp_old, unsigned int size,
      gfp_t gfp_extra_flags)
{
 gfp_t gfp_flags = GFP_KERNEL | __GFP_HIGHMEM | __GFP_ZERO |
     gfp_extra_flags;
 struct bpf_prog *fp;

 BUG_ON(fp_old == NULL);

 size = round_up(size, PAGE_SIZE);
 if (size <= fp_old->pages * PAGE_SIZE)
  return fp_old;

 fp = __vmalloc(size, gfp_flags, PAGE_KERNEL);
 if (fp != NULL) {
  kmemcheck_annotate_bitfield(fp, meta);

  memcpy(fp, fp_old, fp_old->pages * PAGE_SIZE);
  fp->pages = size / PAGE_SIZE;
  fp->aux->prog = fp;




  fp_old->aux = NULL;
  __bpf_prog_free(fp_old);
 }

 return fp;
}

void __bpf_prog_free(struct bpf_prog *fp)
{
 kfree(fp->aux);
 vfree(fp);
}

static bool bpf_is_jmp_and_has_target(const struct bpf_insn *insn)
{
 return BPF_CLASS(insn->code) == BPF_JMP &&



        BPF_OP(insn->code) != BPF_CALL &&
        BPF_OP(insn->code) != BPF_EXIT;
}

static void bpf_adj_branches(struct bpf_prog *prog, u32 pos, u32 delta)
{
 struct bpf_insn *insn = prog->insnsi;
 u32 i, insn_cnt = prog->len;

 for (i = 0; i < insn_cnt; i++, insn++) {
  if (!bpf_is_jmp_and_has_target(insn))
   continue;


  if (i < pos && i + insn->off + 1 > pos)
   insn->off += delta;
  else if (i > pos + delta && i + insn->off + 1 <= pos + delta)
   insn->off -= delta;
 }
}

struct bpf_prog *bpf_patch_insn_single(struct bpf_prog *prog, u32 off,
           const struct bpf_insn *patch, u32 len)
{
 u32 insn_adj_cnt, insn_rest, insn_delta = len - 1;
 struct bpf_prog *prog_adj;


 if (insn_delta == 0) {
  memcpy(prog->insnsi + off, patch, sizeof(*patch));
  return prog;
 }

 insn_adj_cnt = prog->len + insn_delta;





 prog_adj = bpf_prog_realloc(prog, bpf_prog_size(insn_adj_cnt),
        GFP_USER);
 if (!prog_adj)
  return NULL;

 prog_adj->len = insn_adj_cnt;
 insn_rest = insn_adj_cnt - off - len;

 memmove(prog_adj->insnsi + off + len, prog_adj->insnsi + off + 1,
  sizeof(*patch) * insn_rest);
 memcpy(prog_adj->insnsi + off, patch, sizeof(*patch) * len);

 bpf_adj_branches(prog_adj, off, insn_delta);

 return prog_adj;
}

struct bpf_binary_header *
bpf_jit_binary_alloc(unsigned int proglen, u8 **image_ptr,
       unsigned int alignment,
       bpf_jit_fill_hole_t bpf_fill_ill_insns)
{
 struct bpf_binary_header *hdr;
 unsigned int size, hole, start;





 size = round_up(proglen + sizeof(*hdr) + 128, PAGE_SIZE);
 hdr = module_alloc(size);
 if (hdr == NULL)
  return NULL;


 bpf_fill_ill_insns(hdr, size);

 hdr->pages = size / PAGE_SIZE;
 hole = min_t(unsigned int, size - (proglen + sizeof(*hdr)),
       PAGE_SIZE - sizeof(*hdr));
 start = (get_random_int() % hole) & ~(alignment - 1);


 *image_ptr = &hdr->image[start];

 return hdr;
}

void bpf_jit_binary_free(struct bpf_binary_header *hdr)
{
 module_memfree(hdr);
}

int bpf_jit_harden __read_mostly;

static int bpf_jit_blind_insn(const struct bpf_insn *from,
         const struct bpf_insn *aux,
         struct bpf_insn *to_buff)
{
 struct bpf_insn *to = to_buff;
 u32 imm_rnd = get_random_int();
 s16 off;

 BUILD_BUG_ON(BPF_REG_AX + 1 != MAX_BPF_JIT_REG);
 BUILD_BUG_ON(MAX_BPF_REG + 1 != MAX_BPF_JIT_REG);

 if (from->imm == 0 &&
     (from->code == (BPF_ALU | BPF_MOV | BPF_K) ||
      from->code == (BPF_ALU64 | BPF_MOV | BPF_K))) {
  *to++ = BPF_ALU64_REG(BPF_XOR, from->dst_reg, from->dst_reg);
  goto out;
 }

 switch (from->code) {
 case BPF_ALU | BPF_ADD | BPF_K:
 case BPF_ALU | BPF_SUB | BPF_K:
 case BPF_ALU | BPF_AND | BPF_K:
 case BPF_ALU | BPF_OR | BPF_K:
 case BPF_ALU | BPF_XOR | BPF_K:
 case BPF_ALU | BPF_MUL | BPF_K:
 case BPF_ALU | BPF_MOV | BPF_K:
 case BPF_ALU | BPF_DIV | BPF_K:
 case BPF_ALU | BPF_MOD | BPF_K:
  *to++ = BPF_ALU32_IMM(BPF_MOV, BPF_REG_AX, imm_rnd ^ from->imm);
  *to++ = BPF_ALU32_IMM(BPF_XOR, BPF_REG_AX, imm_rnd);
  *to++ = BPF_ALU32_REG(from->code, from->dst_reg, BPF_REG_AX);
  break;

 case BPF_ALU64 | BPF_ADD | BPF_K:
 case BPF_ALU64 | BPF_SUB | BPF_K:
 case BPF_ALU64 | BPF_AND | BPF_K:
 case BPF_ALU64 | BPF_OR | BPF_K:
 case BPF_ALU64 | BPF_XOR | BPF_K:
 case BPF_ALU64 | BPF_MUL | BPF_K:
 case BPF_ALU64 | BPF_MOV | BPF_K:
 case BPF_ALU64 | BPF_DIV | BPF_K:
 case BPF_ALU64 | BPF_MOD | BPF_K:
  *to++ = BPF_ALU64_IMM(BPF_MOV, BPF_REG_AX, imm_rnd ^ from->imm);
  *to++ = BPF_ALU64_IMM(BPF_XOR, BPF_REG_AX, imm_rnd);
  *to++ = BPF_ALU64_REG(from->code, from->dst_reg, BPF_REG_AX);
  break;

 case BPF_JMP | BPF_JEQ | BPF_K:
 case BPF_JMP | BPF_JNE | BPF_K:
 case BPF_JMP | BPF_JGT | BPF_K:
 case BPF_JMP | BPF_JGE | BPF_K:
 case BPF_JMP | BPF_JSGT | BPF_K:
 case BPF_JMP | BPF_JSGE | BPF_K:
 case BPF_JMP | BPF_JSET | BPF_K:

  off = from->off;
  if (off < 0)
   off -= 2;
  *to++ = BPF_ALU64_IMM(BPF_MOV, BPF_REG_AX, imm_rnd ^ from->imm);
  *to++ = BPF_ALU64_IMM(BPF_XOR, BPF_REG_AX, imm_rnd);
  *to++ = BPF_JMP_REG(from->code, from->dst_reg, BPF_REG_AX, off);
  break;

 case BPF_LD | BPF_ABS | BPF_W:
 case BPF_LD | BPF_ABS | BPF_H:
 case BPF_LD | BPF_ABS | BPF_B:
  *to++ = BPF_ALU64_IMM(BPF_MOV, BPF_REG_AX, imm_rnd ^ from->imm);
  *to++ = BPF_ALU64_IMM(BPF_XOR, BPF_REG_AX, imm_rnd);
  *to++ = BPF_LD_IND(from->code, BPF_REG_AX, 0);
  break;

 case BPF_LD | BPF_IND | BPF_W:
 case BPF_LD | BPF_IND | BPF_H:
 case BPF_LD | BPF_IND | BPF_B:
  *to++ = BPF_ALU64_IMM(BPF_MOV, BPF_REG_AX, imm_rnd ^ from->imm);
  *to++ = BPF_ALU64_IMM(BPF_XOR, BPF_REG_AX, imm_rnd);
  *to++ = BPF_ALU32_REG(BPF_ADD, BPF_REG_AX, from->src_reg);
  *to++ = BPF_LD_IND(from->code, BPF_REG_AX, 0);
  break;

 case BPF_LD | BPF_IMM | BPF_DW:
  *to++ = BPF_ALU64_IMM(BPF_MOV, BPF_REG_AX, imm_rnd ^ aux[1].imm);
  *to++ = BPF_ALU64_IMM(BPF_XOR, BPF_REG_AX, imm_rnd);
  *to++ = BPF_ALU64_IMM(BPF_LSH, BPF_REG_AX, 32);
  *to++ = BPF_ALU64_REG(BPF_MOV, aux[0].dst_reg, BPF_REG_AX);
  break;
 case 0:
  *to++ = BPF_ALU32_IMM(BPF_MOV, BPF_REG_AX, imm_rnd ^ aux[0].imm);
  *to++ = BPF_ALU32_IMM(BPF_XOR, BPF_REG_AX, imm_rnd);
  *to++ = BPF_ALU64_REG(BPF_OR, aux[0].dst_reg, BPF_REG_AX);
  break;

 case BPF_ST | BPF_MEM | BPF_DW:
 case BPF_ST | BPF_MEM | BPF_W:
 case BPF_ST | BPF_MEM | BPF_H:
 case BPF_ST | BPF_MEM | BPF_B:
  *to++ = BPF_ALU64_IMM(BPF_MOV, BPF_REG_AX, imm_rnd ^ from->imm);
  *to++ = BPF_ALU64_IMM(BPF_XOR, BPF_REG_AX, imm_rnd);
  *to++ = BPF_STX_MEM(from->code, from->dst_reg, BPF_REG_AX, from->off);
  break;
 }
out:
 return to - to_buff;
}

static struct bpf_prog *bpf_prog_clone_create(struct bpf_prog *fp_other,
           gfp_t gfp_extra_flags)
{
 gfp_t gfp_flags = GFP_KERNEL | __GFP_HIGHMEM | __GFP_ZERO |
     gfp_extra_flags;
 struct bpf_prog *fp;

 fp = __vmalloc(fp_other->pages * PAGE_SIZE, gfp_flags, PAGE_KERNEL);
 if (fp != NULL) {
  kmemcheck_annotate_bitfield(fp, meta);





  memcpy(fp, fp_other, fp_other->pages * PAGE_SIZE);
 }

 return fp;
}

static void bpf_prog_clone_free(struct bpf_prog *fp)
{







 fp->aux = NULL;
 __bpf_prog_free(fp);
}

void bpf_jit_prog_release_other(struct bpf_prog *fp, struct bpf_prog *fp_other)
{



 fp->aux->prog = fp;
 bpf_prog_clone_free(fp_other);
}

struct bpf_prog *bpf_jit_blind_constants(struct bpf_prog *prog)
{
 struct bpf_insn insn_buff[16], aux[2];
 struct bpf_prog *clone, *tmp;
 int insn_delta, insn_cnt;
 struct bpf_insn *insn;
 int i, rewritten;

 if (!bpf_jit_blinding_enabled())
  return prog;

 clone = bpf_prog_clone_create(prog, GFP_USER);
 if (!clone)
  return ERR_PTR(-ENOMEM);

 insn_cnt = clone->len;
 insn = clone->insnsi;

 for (i = 0; i < insn_cnt; i++, insn++) {




  if (insn[0].code == (BPF_LD | BPF_IMM | BPF_DW) &&
      insn[1].code == 0)
   memcpy(aux, insn, sizeof(aux));

  rewritten = bpf_jit_blind_insn(insn, aux, insn_buff);
  if (!rewritten)
   continue;

  tmp = bpf_patch_insn_single(clone, i, insn_buff, rewritten);
  if (!tmp) {




   bpf_jit_prog_release_other(prog, clone);
   return ERR_PTR(-ENOMEM);
  }

  clone = tmp;
  insn_delta = rewritten - 1;


  insn = clone->insnsi + i + insn_delta;
  insn_cnt += insn_delta;
  i += insn_delta;
 }

 return clone;
}





noinline u64 __bpf_call_base(u64 r1, u64 r2, u64 r3, u64 r4, u64 r5)
{
 return 0;
}
EXPORT_SYMBOL_GPL(__bpf_call_base);
static unsigned int __bpf_prog_run(void *ctx, const struct bpf_insn *insn)
{
 u64 stack[MAX_BPF_STACK / sizeof(u64)];
 u64 regs[MAX_BPF_REG], tmp;
 static const void *jumptable[256] = {
  [0 ... 255] = &&default_label,


  [BPF_ALU | BPF_ADD | BPF_X] = &&ALU_ADD_X,
  [BPF_ALU | BPF_ADD | BPF_K] = &&ALU_ADD_K,
  [BPF_ALU | BPF_SUB | BPF_X] = &&ALU_SUB_X,
  [BPF_ALU | BPF_SUB | BPF_K] = &&ALU_SUB_K,
  [BPF_ALU | BPF_AND | BPF_X] = &&ALU_AND_X,
  [BPF_ALU | BPF_AND | BPF_K] = &&ALU_AND_K,
  [BPF_ALU | BPF_OR | BPF_X] = &&ALU_OR_X,
  [BPF_ALU | BPF_OR | BPF_K] = &&ALU_OR_K,
  [BPF_ALU | BPF_LSH | BPF_X] = &&ALU_LSH_X,
  [BPF_ALU | BPF_LSH | BPF_K] = &&ALU_LSH_K,
  [BPF_ALU | BPF_RSH | BPF_X] = &&ALU_RSH_X,
  [BPF_ALU | BPF_RSH | BPF_K] = &&ALU_RSH_K,
  [BPF_ALU | BPF_XOR | BPF_X] = &&ALU_XOR_X,
  [BPF_ALU | BPF_XOR | BPF_K] = &&ALU_XOR_K,
  [BPF_ALU | BPF_MUL | BPF_X] = &&ALU_MUL_X,
  [BPF_ALU | BPF_MUL | BPF_K] = &&ALU_MUL_K,
  [BPF_ALU | BPF_MOV | BPF_X] = &&ALU_MOV_X,
  [BPF_ALU | BPF_MOV | BPF_K] = &&ALU_MOV_K,
  [BPF_ALU | BPF_DIV | BPF_X] = &&ALU_DIV_X,
  [BPF_ALU | BPF_DIV | BPF_K] = &&ALU_DIV_K,
  [BPF_ALU | BPF_MOD | BPF_X] = &&ALU_MOD_X,
  [BPF_ALU | BPF_MOD | BPF_K] = &&ALU_MOD_K,
  [BPF_ALU | BPF_NEG] = &&ALU_NEG,
  [BPF_ALU | BPF_END | BPF_TO_BE] = &&ALU_END_TO_BE,
  [BPF_ALU | BPF_END | BPF_TO_LE] = &&ALU_END_TO_LE,

  [BPF_ALU64 | BPF_ADD | BPF_X] = &&ALU64_ADD_X,
  [BPF_ALU64 | BPF_ADD | BPF_K] = &&ALU64_ADD_K,
  [BPF_ALU64 | BPF_SUB | BPF_X] = &&ALU64_SUB_X,
  [BPF_ALU64 | BPF_SUB | BPF_K] = &&ALU64_SUB_K,
  [BPF_ALU64 | BPF_AND | BPF_X] = &&ALU64_AND_X,
  [BPF_ALU64 | BPF_AND | BPF_K] = &&ALU64_AND_K,
  [BPF_ALU64 | BPF_OR | BPF_X] = &&ALU64_OR_X,
  [BPF_ALU64 | BPF_OR | BPF_K] = &&ALU64_OR_K,
  [BPF_ALU64 | BPF_LSH | BPF_X] = &&ALU64_LSH_X,
  [BPF_ALU64 | BPF_LSH | BPF_K] = &&ALU64_LSH_K,
  [BPF_ALU64 | BPF_RSH | BPF_X] = &&ALU64_RSH_X,
  [BPF_ALU64 | BPF_RSH | BPF_K] = &&ALU64_RSH_K,
  [BPF_ALU64 | BPF_XOR | BPF_X] = &&ALU64_XOR_X,
  [BPF_ALU64 | BPF_XOR | BPF_K] = &&ALU64_XOR_K,
  [BPF_ALU64 | BPF_MUL | BPF_X] = &&ALU64_MUL_X,
  [BPF_ALU64 | BPF_MUL | BPF_K] = &&ALU64_MUL_K,
  [BPF_ALU64 | BPF_MOV | BPF_X] = &&ALU64_MOV_X,
  [BPF_ALU64 | BPF_MOV | BPF_K] = &&ALU64_MOV_K,
  [BPF_ALU64 | BPF_ARSH | BPF_X] = &&ALU64_ARSH_X,
  [BPF_ALU64 | BPF_ARSH | BPF_K] = &&ALU64_ARSH_K,
  [BPF_ALU64 | BPF_DIV | BPF_X] = &&ALU64_DIV_X,
  [BPF_ALU64 | BPF_DIV | BPF_K] = &&ALU64_DIV_K,
  [BPF_ALU64 | BPF_MOD | BPF_X] = &&ALU64_MOD_X,
  [BPF_ALU64 | BPF_MOD | BPF_K] = &&ALU64_MOD_K,
  [BPF_ALU64 | BPF_NEG] = &&ALU64_NEG,

  [BPF_JMP | BPF_CALL] = &&JMP_CALL,
  [BPF_JMP | BPF_CALL | BPF_X] = &&JMP_TAIL_CALL,

  [BPF_JMP | BPF_JA] = &&JMP_JA,
  [BPF_JMP | BPF_JEQ | BPF_X] = &&JMP_JEQ_X,
  [BPF_JMP | BPF_JEQ | BPF_K] = &&JMP_JEQ_K,
  [BPF_JMP | BPF_JNE | BPF_X] = &&JMP_JNE_X,
  [BPF_JMP | BPF_JNE | BPF_K] = &&JMP_JNE_K,
  [BPF_JMP | BPF_JGT | BPF_X] = &&JMP_JGT_X,
  [BPF_JMP | BPF_JGT | BPF_K] = &&JMP_JGT_K,
  [BPF_JMP | BPF_JGE | BPF_X] = &&JMP_JGE_X,
  [BPF_JMP | BPF_JGE | BPF_K] = &&JMP_JGE_K,
  [BPF_JMP | BPF_JSGT | BPF_X] = &&JMP_JSGT_X,
  [BPF_JMP | BPF_JSGT | BPF_K] = &&JMP_JSGT_K,
  [BPF_JMP | BPF_JSGE | BPF_X] = &&JMP_JSGE_X,
  [BPF_JMP | BPF_JSGE | BPF_K] = &&JMP_JSGE_K,
  [BPF_JMP | BPF_JSET | BPF_X] = &&JMP_JSET_X,
  [BPF_JMP | BPF_JSET | BPF_K] = &&JMP_JSET_K,

  [BPF_JMP | BPF_EXIT] = &&JMP_EXIT,

  [BPF_STX | BPF_MEM | BPF_B] = &&STX_MEM_B,
  [BPF_STX | BPF_MEM | BPF_H] = &&STX_MEM_H,
  [BPF_STX | BPF_MEM | BPF_W] = &&STX_MEM_W,
  [BPF_STX | BPF_MEM | BPF_DW] = &&STX_MEM_DW,
  [BPF_STX | BPF_XADD | BPF_W] = &&STX_XADD_W,
  [BPF_STX | BPF_XADD | BPF_DW] = &&STX_XADD_DW,
  [BPF_ST | BPF_MEM | BPF_B] = &&ST_MEM_B,
  [BPF_ST | BPF_MEM | BPF_H] = &&ST_MEM_H,
  [BPF_ST | BPF_MEM | BPF_W] = &&ST_MEM_W,
  [BPF_ST | BPF_MEM | BPF_DW] = &&ST_MEM_DW,

  [BPF_LDX | BPF_MEM | BPF_B] = &&LDX_MEM_B,
  [BPF_LDX | BPF_MEM | BPF_H] = &&LDX_MEM_H,
  [BPF_LDX | BPF_MEM | BPF_W] = &&LDX_MEM_W,
  [BPF_LDX | BPF_MEM | BPF_DW] = &&LDX_MEM_DW,
  [BPF_LD | BPF_ABS | BPF_W] = &&LD_ABS_W,
  [BPF_LD | BPF_ABS | BPF_H] = &&LD_ABS_H,
  [BPF_LD | BPF_ABS | BPF_B] = &&LD_ABS_B,
  [BPF_LD | BPF_IND | BPF_W] = &&LD_IND_W,
  [BPF_LD | BPF_IND | BPF_H] = &&LD_IND_H,
  [BPF_LD | BPF_IND | BPF_B] = &&LD_IND_B,
  [BPF_LD | BPF_IMM | BPF_DW] = &&LD_IMM_DW,
 };
 u32 tail_call_cnt = 0;
 void *ptr;
 int off;


 FP = (u64) (unsigned long) &stack[ARRAY_SIZE(stack)];
 ARG1 = (u64) (unsigned long) ctx;

select_insn:
 goto *jumptable[insn->code];


 ALU64_##OPCODE##_X: \
  DST = DST OP SRC; \
  CONT; \
 ALU_##OPCODE##_X: \
  DST = (u32) DST OP (u32) SRC; \
  CONT; \
 ALU64_##OPCODE##_K: \
  DST = DST OP IMM; \
  CONT; \
 ALU_##OPCODE##_K: \
  DST = (u32) DST OP (u32) IMM; \
  CONT;

 ALU(ADD, +)
 ALU(SUB, -)
 ALU(AND, &)
 ALU(OR, |)
 ALU(LSH, <<)
 ALU(RSH, >>)
 ALU(XOR, ^)
 ALU(MUL, *)
 ALU_NEG:
  DST = (u32) -DST;
  CONT;
 ALU64_NEG:
  DST = -DST;
  CONT;
 ALU_MOV_X:
  DST = (u32) SRC;
  CONT;
 ALU_MOV_K:
  DST = (u32) IMM;
  CONT;
 ALU64_MOV_X:
  DST = SRC;
  CONT;
 ALU64_MOV_K:
  DST = IMM;
  CONT;
 LD_IMM_DW:
  DST = (u64) (u32) insn[0].imm | ((u64) (u32) insn[1].imm) << 32;
  insn++;
  CONT;
 ALU64_ARSH_X:
  (*(s64 *) &DST) >>= SRC;
  CONT;
 ALU64_ARSH_K:
  (*(s64 *) &DST) >>= IMM;
  CONT;
 ALU64_MOD_X:
  if (unlikely(SRC == 0))
   return 0;
  div64_u64_rem(DST, SRC, &tmp);
  DST = tmp;
  CONT;
 ALU_MOD_X:
  if (unlikely(SRC == 0))
   return 0;
  tmp = (u32) DST;
  DST = do_div(tmp, (u32) SRC);
  CONT;
 ALU64_MOD_K:
  div64_u64_rem(DST, IMM, &tmp);
  DST = tmp;
  CONT;
 ALU_MOD_K:
  tmp = (u32) DST;
  DST = do_div(tmp, (u32) IMM);
  CONT;
 ALU64_DIV_X:
  if (unlikely(SRC == 0))
   return 0;
  DST = div64_u64(DST, SRC);
  CONT;
 ALU_DIV_X:
  if (unlikely(SRC == 0))
   return 0;
  tmp = (u32) DST;
  do_div(tmp, (u32) SRC);
  DST = (u32) tmp;
  CONT;
 ALU64_DIV_K:
  DST = div64_u64(DST, IMM);
  CONT;
 ALU_DIV_K:
  tmp = (u32) DST;
  do_div(tmp, (u32) IMM);
  DST = (u32) tmp;
  CONT;
 ALU_END_TO_BE:
  switch (IMM) {
  case 16:
   DST = (__force u16) cpu_to_be16(DST);
   break;
  case 32:
   DST = (__force u32) cpu_to_be32(DST);
   break;
  case 64:
   DST = (__force u64) cpu_to_be64(DST);
   break;
  }
  CONT;
 ALU_END_TO_LE:
  switch (IMM) {
  case 16:
   DST = (__force u16) cpu_to_le16(DST);
   break;
  case 32:
   DST = (__force u32) cpu_to_le32(DST);
   break;
  case 64:
   DST = (__force u64) cpu_to_le64(DST);
   break;
  }
  CONT;


 JMP_CALL:




  BPF_R0 = (__bpf_call_base + insn->imm)(BPF_R1, BPF_R2, BPF_R3,
             BPF_R4, BPF_R5);
  CONT;

 JMP_TAIL_CALL: {
  struct bpf_map *map = (struct bpf_map *) (unsigned long) BPF_R2;
  struct bpf_array *array = container_of(map, struct bpf_array, map);
  struct bpf_prog *prog;
  u64 index = BPF_R3;

  if (unlikely(index >= array->map.max_entries))
   goto out;

  if (unlikely(tail_call_cnt > MAX_TAIL_CALL_CNT))
   goto out;

  tail_call_cnt++;

  prog = READ_ONCE(array->ptrs[index]);
  if (unlikely(!prog))
   goto out;






  insn = prog->insnsi;
  goto select_insn;
out:
  CONT;
 }

 JMP_JA:
  insn += insn->off;
  CONT;
 JMP_JEQ_X:
  if (DST == SRC) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_JEQ_K:
  if (DST == IMM) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_JNE_X:
  if (DST != SRC) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_JNE_K:
  if (DST != IMM) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_JGT_X:
  if (DST > SRC) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_JGT_K:
  if (DST > IMM) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_JGE_X:
  if (DST >= SRC) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_JGE_K:
  if (DST >= IMM) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_JSGT_X:
  if (((s64) DST) > ((s64) SRC)) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_JSGT_K:
  if (((s64) DST) > ((s64) IMM)) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_JSGE_X:
  if (((s64) DST) >= ((s64) SRC)) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_JSGE_K:
  if (((s64) DST) >= ((s64) IMM)) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_JSET_X:
  if (DST & SRC) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_JSET_K:
  if (DST & IMM) {
   insn += insn->off;
   CONT_JMP;
  }
  CONT;
 JMP_EXIT:
  return BPF_R0;


 STX_MEM_##SIZEOP: \
  *(SIZE *)(unsigned long) (DST + insn->off) = SRC; \
  CONT; \
 ST_MEM_##SIZEOP: \
  *(SIZE *)(unsigned long) (DST + insn->off) = IMM; \
  CONT; \
 LDX_MEM_##SIZEOP: \
  DST = *(SIZE *)(unsigned long) (SRC + insn->off); \
  CONT;

 LDST(B, u8)
 LDST(H, u16)
 LDST(W, u32)
 LDST(DW, u64)
 STX_XADD_W:
  atomic_add((u32) SRC, (atomic_t *)(unsigned long)
      (DST + insn->off));
  CONT;
 STX_XADD_DW:
  atomic64_add((u64) SRC, (atomic64_t *)(unsigned long)
        (DST + insn->off));
  CONT;
 LD_ABS_W:
  off = IMM;
load_word:
  ptr = bpf_load_pointer((struct sk_buff *) (unsigned long) CTX, off, 4, &tmp);
  if (likely(ptr != NULL)) {
   BPF_R0 = get_unaligned_be32(ptr);
   CONT;
  }

  return 0;
 LD_ABS_H:
  off = IMM;
load_half:
  ptr = bpf_load_pointer((struct sk_buff *) (unsigned long) CTX, off, 2, &tmp);
  if (likely(ptr != NULL)) {
   BPF_R0 = get_unaligned_be16(ptr);
   CONT;
  }

  return 0;
 LD_ABS_B:
  off = IMM;
load_byte:
  ptr = bpf_load_pointer((struct sk_buff *) (unsigned long) CTX, off, 1, &tmp);
  if (likely(ptr != NULL)) {
   BPF_R0 = *(u8 *)ptr;
   CONT;
  }

  return 0;
 LD_IND_W:
  off = IMM + SRC;
  goto load_word;
 LD_IND_H:
  off = IMM + SRC;
  goto load_half;
 LD_IND_B:
  off = IMM + SRC;
  goto load_byte;

 default_label:

  WARN_RATELIMIT(1, "unknown opcode %02x\n", insn->code);
  return 0;
}
STACK_FRAME_NON_STANDARD(__bpf_prog_run);

bool bpf_prog_array_compatible(struct bpf_array *array,
          const struct bpf_prog *fp)
{
 if (!array->owner_prog_type) {



  array->owner_prog_type = fp->type;
  array->owner_jited = fp->jited;

  return true;
 }

 return array->owner_prog_type == fp->type &&
        array->owner_jited == fp->jited;
}

static int bpf_check_tail_call(const struct bpf_prog *fp)
{
 struct bpf_prog_aux *aux = fp->aux;
 int i;

 for (i = 0; i < aux->used_map_cnt; i++) {
  struct bpf_map *map = aux->used_maps[i];
  struct bpf_array *array;

  if (map->map_type != BPF_MAP_TYPE_PROG_ARRAY)
   continue;

  array = container_of(map, struct bpf_array, map);
  if (!bpf_prog_array_compatible(array, fp))
   return -EINVAL;
 }

 return 0;
}
struct bpf_prog *bpf_prog_select_runtime(struct bpf_prog *fp, int *err)
{
 fp->bpf_func = (void *) __bpf_prog_run;







 fp = bpf_int_jit_compile(fp);
 bpf_prog_lock_ro(fp);






 *err = bpf_check_tail_call(fp);

 return fp;
}
EXPORT_SYMBOL_GPL(bpf_prog_select_runtime);

static void bpf_prog_free_deferred(struct work_struct *work)
{
 struct bpf_prog_aux *aux;

 aux = container_of(work, struct bpf_prog_aux, work);
 bpf_jit_free(aux->prog);
}


void bpf_prog_free(struct bpf_prog *fp)
{
 struct bpf_prog_aux *aux = fp->aux;

 INIT_WORK(&aux->work, bpf_prog_free_deferred);
 schedule_work(&aux->work);
}
EXPORT_SYMBOL_GPL(bpf_prog_free);


static DEFINE_PER_CPU(struct rnd_state, bpf_user_rnd_state);

void bpf_user_rnd_init_once(void)
{
 prandom_init_once(&bpf_user_rnd_state);
}

u64 bpf_user_rnd_u32(u64 r1, u64 r2, u64 r3, u64 r4, u64 r5)
{






 struct rnd_state *state;
 u32 res;

 state = &get_cpu_var(bpf_user_rnd_state);
 res = prandom_u32_state(state);
 put_cpu_var(state);

 return res;
}


const struct bpf_func_proto bpf_map_lookup_elem_proto __weak;
const struct bpf_func_proto bpf_map_update_elem_proto __weak;
const struct bpf_func_proto bpf_map_delete_elem_proto __weak;

const struct bpf_func_proto bpf_get_prandom_u32_proto __weak;
const struct bpf_func_proto bpf_get_smp_processor_id_proto __weak;
const struct bpf_func_proto bpf_ktime_get_ns_proto __weak;

const struct bpf_func_proto bpf_get_current_pid_tgid_proto __weak;
const struct bpf_func_proto bpf_get_current_uid_gid_proto __weak;
const struct bpf_func_proto bpf_get_current_comm_proto __weak;

const struct bpf_func_proto * __weak bpf_get_trace_printk_proto(void)
{
 return NULL;
}

const struct bpf_func_proto * __weak bpf_get_event_output_proto(void)
{
 return NULL;
}


const struct bpf_func_proto bpf_tail_call_proto = {
 .func = NULL,
 .gpl_only = false,
 .ret_type = RET_VOID,
 .arg1_type = ARG_PTR_TO_CTX,
 .arg2_type = ARG_CONST_MAP_PTR,
 .arg3_type = ARG_ANYTHING,
};


struct bpf_prog * __weak bpf_int_jit_compile(struct bpf_prog *prog)
{
 return prog;
}

bool __weak bpf_helper_changes_skb_data(void *func)
{
 return false;
}




int __weak skb_copy_bits(const struct sk_buff *skb, int offset, void *to,
    int len)
{
 return -EFAULT;
}







struct cpuhp_cpu_state {
 enum cpuhp_state state;
 enum cpuhp_state target;
 struct task_struct *thread;
 bool should_run;
 bool rollback;
 enum cpuhp_state cb_state;
 int (*cb)(unsigned int cpu);
 int result;
 struct completion done;
};

static DEFINE_PER_CPU(struct cpuhp_cpu_state, cpuhp_state);
struct cpuhp_step {
 const char *name;
 int (*startup)(unsigned int cpu);
 int (*teardown)(unsigned int cpu);
 bool skip_onerr;
 bool cant_stop;
};

static DEFINE_MUTEX(cpuhp_state_mutex);
static struct cpuhp_step cpuhp_bp_states[];
static struct cpuhp_step cpuhp_ap_states[];
static int cpuhp_invoke_callback(unsigned int cpu, enum cpuhp_state step,
     int (*cb)(unsigned int))
{
 struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, cpu);
 int ret = 0;

 if (cb) {
  trace_cpuhp_enter(cpu, st->target, step, cb);
  ret = cb(cpu);
  trace_cpuhp_exit(cpu, st->state, step, ret);
 }
 return ret;
}


static DEFINE_MUTEX(cpu_add_remove_lock);
bool cpuhp_tasks_frozen;
EXPORT_SYMBOL_GPL(cpuhp_tasks_frozen);
void cpu_maps_update_begin(void)
{
 mutex_lock(&cpu_add_remove_lock);
}
EXPORT_SYMBOL(cpu_notifier_register_begin);

void cpu_maps_update_done(void)
{
 mutex_unlock(&cpu_add_remove_lock);
}
EXPORT_SYMBOL(cpu_notifier_register_done);

static RAW_NOTIFIER_HEAD(cpu_chain);




static int cpu_hotplug_disabled;


static struct {
 struct task_struct *active_writer;

 wait_queue_head_t wq;

 struct mutex lock;




 atomic_t refcount;

 struct lockdep_map dep_map;
} cpu_hotplug = {
 .active_writer = NULL,
 .wq = __WAIT_QUEUE_HEAD_INITIALIZER(cpu_hotplug.wq),
 .lock = __MUTEX_INITIALIZER(cpu_hotplug.lock),
 .dep_map = {.name = "cpu_hotplug.lock" },
};


      lock_map_acquire_tryread(&cpu_hotplug.dep_map)


void get_online_cpus(void)
{
 might_sleep();
 if (cpu_hotplug.active_writer == current)
  return;
 cpuhp_lock_acquire_read();
 mutex_lock(&cpu_hotplug.lock);
 atomic_inc(&cpu_hotplug.refcount);
 mutex_unlock(&cpu_hotplug.lock);
}
EXPORT_SYMBOL_GPL(get_online_cpus);

void put_online_cpus(void)
{
 int refcount;

 if (cpu_hotplug.active_writer == current)
  return;

 refcount = atomic_dec_return(&cpu_hotplug.refcount);
 if (WARN_ON(refcount < 0))
  atomic_inc(&cpu_hotplug.refcount);

 if (refcount <= 0 && waitqueue_active(&cpu_hotplug.wq))
  wake_up(&cpu_hotplug.wq);

 cpuhp_lock_release();

}
EXPORT_SYMBOL_GPL(put_online_cpus);
void cpu_hotplug_begin(void)
{
 DEFINE_WAIT(wait);

 cpu_hotplug.active_writer = current;
 cpuhp_lock_acquire();

 for (;;) {
  mutex_lock(&cpu_hotplug.lock);
  prepare_to_wait(&cpu_hotplug.wq, &wait, TASK_UNINTERRUPTIBLE);
  if (likely(!atomic_read(&cpu_hotplug.refcount)))
    break;
  mutex_unlock(&cpu_hotplug.lock);
  schedule();
 }
 finish_wait(&cpu_hotplug.wq, &wait);
}

void cpu_hotplug_done(void)
{
 cpu_hotplug.active_writer = NULL;
 mutex_unlock(&cpu_hotplug.lock);
 cpuhp_lock_release();
}
void cpu_hotplug_disable(void)
{
 cpu_maps_update_begin();
 cpu_hotplug_disabled++;
 cpu_maps_update_done();
}
EXPORT_SYMBOL_GPL(cpu_hotplug_disable);

void cpu_hotplug_enable(void)
{
 cpu_maps_update_begin();
 WARN_ON(--cpu_hotplug_disabled < 0);
 cpu_maps_update_done();
}
EXPORT_SYMBOL_GPL(cpu_hotplug_enable);


int register_cpu_notifier(struct notifier_block *nb)
{
 int ret;
 cpu_maps_update_begin();
 ret = raw_notifier_chain_register(&cpu_chain, nb);
 cpu_maps_update_done();
 return ret;
}

int __register_cpu_notifier(struct notifier_block *nb)
{
 return raw_notifier_chain_register(&cpu_chain, nb);
}

static int __cpu_notify(unsigned long val, unsigned int cpu, int nr_to_call,
   int *nr_calls)
{
 unsigned long mod = cpuhp_tasks_frozen ? CPU_TASKS_FROZEN : 0;
 void *hcpu = (void *)(long)cpu;

 int ret;

 ret = __raw_notifier_call_chain(&cpu_chain, val | mod, hcpu, nr_to_call,
     nr_calls);

 return notifier_to_errno(ret);
}

static int cpu_notify(unsigned long val, unsigned int cpu)
{
 return __cpu_notify(val, cpu, -1, NULL);
}

static void cpu_notify_nofail(unsigned long val, unsigned int cpu)
{
 BUG_ON(cpu_notify(val, cpu));
}


static int notify_prepare(unsigned int cpu)
{
 int nr_calls = 0;
 int ret;

 ret = __cpu_notify(CPU_UP_PREPARE, cpu, -1, &nr_calls);
 if (ret) {
  nr_calls--;
  printk(KERN_WARNING "%s: attempt to bring up CPU %u failed\n",
    __func__, cpu);
  __cpu_notify(CPU_UP_CANCELED, cpu, nr_calls, NULL);
 }
 return ret;
}

static int notify_online(unsigned int cpu)
{
 cpu_notify(CPU_ONLINE, cpu);
 return 0;
}

static int notify_starting(unsigned int cpu)
{
 cpu_notify(CPU_STARTING, cpu);
 return 0;
}

static int bringup_wait_for_ap(unsigned int cpu)
{
 struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, cpu);

 wait_for_completion(&st->done);
 return st->result;
}

static int bringup_cpu(unsigned int cpu)
{
 struct task_struct *idle = idle_thread_get(cpu);
 int ret;


 ret = __cpu_up(cpu, idle);
 if (ret) {
  cpu_notify(CPU_UP_CANCELED, cpu);
  return ret;
 }
 ret = bringup_wait_for_ap(cpu);
 BUG_ON(!cpu_online(cpu));
 return ret;
}




static void undo_cpu_down(unsigned int cpu, struct cpuhp_cpu_state *st,
     struct cpuhp_step *steps)
{
 for (st->state++; st->state < st->target; st->state++) {
  struct cpuhp_step *step = steps + st->state;

  if (!step->skip_onerr)
   cpuhp_invoke_callback(cpu, st->state, step->startup);
 }
}

static int cpuhp_down_callbacks(unsigned int cpu, struct cpuhp_cpu_state *st,
    struct cpuhp_step *steps, enum cpuhp_state target)
{
 enum cpuhp_state prev_state = st->state;
 int ret = 0;

 for (; st->state > target; st->state--) {
  struct cpuhp_step *step = steps + st->state;

  ret = cpuhp_invoke_callback(cpu, st->state, step->teardown);
  if (ret) {
   st->target = prev_state;
   undo_cpu_down(cpu, st, steps);
   break;
  }
 }
 return ret;
}

static void undo_cpu_up(unsigned int cpu, struct cpuhp_cpu_state *st,
   struct cpuhp_step *steps)
{
 for (st->state--; st->state > st->target; st->state--) {
  struct cpuhp_step *step = steps + st->state;

  if (!step->skip_onerr)
   cpuhp_invoke_callback(cpu, st->state, step->teardown);
 }
}

static int cpuhp_up_callbacks(unsigned int cpu, struct cpuhp_cpu_state *st,
         struct cpuhp_step *steps, enum cpuhp_state target)
{
 enum cpuhp_state prev_state = st->state;
 int ret = 0;

 while (st->state < target) {
  struct cpuhp_step *step;

  st->state++;
  step = steps + st->state;
  ret = cpuhp_invoke_callback(cpu, st->state, step->startup);
  if (ret) {
   st->target = prev_state;
   undo_cpu_up(cpu, st, steps);
   break;
  }
 }
 return ret;
}




static void cpuhp_create(unsigned int cpu)
{
 struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, cpu);

 init_completion(&st->done);
}

static int cpuhp_should_run(unsigned int cpu)
{
 struct cpuhp_cpu_state *st = this_cpu_ptr(&cpuhp_state);

 return st->should_run;
}


static int cpuhp_ap_offline(unsigned int cpu, struct cpuhp_cpu_state *st)
{
 enum cpuhp_state target = max((int)st->target, CPUHP_TEARDOWN_CPU);

 return cpuhp_down_callbacks(cpu, st, cpuhp_ap_states, target);
}


static int cpuhp_ap_online(unsigned int cpu, struct cpuhp_cpu_state *st)
{
 return cpuhp_up_callbacks(cpu, st, cpuhp_ap_states, st->target);
}





static void cpuhp_thread_fun(unsigned int cpu)
{
 struct cpuhp_cpu_state *st = this_cpu_ptr(&cpuhp_state);
 int ret = 0;





 smp_mb();
 if (!st->should_run)
  return;

 st->should_run = false;


 if (st->cb) {
  if (st->cb_state < CPUHP_AP_ONLINE) {
   local_irq_disable();
   ret = cpuhp_invoke_callback(cpu, st->cb_state, st->cb);
   local_irq_enable();
  } else {
   ret = cpuhp_invoke_callback(cpu, st->cb_state, st->cb);
  }
 } else if (st->rollback) {
  BUG_ON(st->state < CPUHP_AP_ONLINE_IDLE);

  undo_cpu_down(cpu, st, cpuhp_ap_states);




  cpu_notify_nofail(CPU_DOWN_FAILED, cpu);
  st->rollback = false;
 } else {

  BUG_ON(st->state < CPUHP_AP_ONLINE_IDLE);


  if (st->state < st->target)
   ret = cpuhp_ap_online(cpu, st);
  else if (st->state > st->target)
   ret = cpuhp_ap_offline(cpu, st);
 }
 st->result = ret;
 complete(&st->done);
}


static int cpuhp_invoke_ap_callback(int cpu, enum cpuhp_state state,
        int (*cb)(unsigned int))
{
 struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, cpu);

 if (!cpu_online(cpu))
  return 0;

 st->cb_state = state;
 st->cb = cb;




 smp_mb();
 st->should_run = true;
 wake_up_process(st->thread);
 wait_for_completion(&st->done);
 return st->result;
}


static void __cpuhp_kick_ap_work(struct cpuhp_cpu_state *st)
{
 st->result = 0;
 st->cb = NULL;




 smp_mb();
 st->should_run = true;
 wake_up_process(st->thread);
}

static int cpuhp_kick_ap_work(unsigned int cpu)
{
 struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, cpu);
 enum cpuhp_state state = st->state;

 trace_cpuhp_enter(cpu, st->target, state, cpuhp_kick_ap_work);
 __cpuhp_kick_ap_work(st);
 wait_for_completion(&st->done);
 trace_cpuhp_exit(cpu, st->state, state, st->result);
 return st->result;
}

static struct smp_hotplug_thread cpuhp_threads = {
 .store = &cpuhp_state.thread,
 .create = &cpuhp_create,
 .thread_should_run = cpuhp_should_run,
 .thread_fn = cpuhp_thread_fun,
 .thread_comm = "cpuhp/%u",
 .selfparking = true,
};

void __init cpuhp_threads_init(void)
{
 BUG_ON(smpboot_register_percpu_thread(&cpuhp_threads));
 kthread_unpark(this_cpu_read(cpuhp_state.thread));
}

EXPORT_SYMBOL(register_cpu_notifier);
EXPORT_SYMBOL(__register_cpu_notifier);
void unregister_cpu_notifier(struct notifier_block *nb)
{
 cpu_maps_update_begin();
 raw_notifier_chain_unregister(&cpu_chain, nb);
 cpu_maps_update_done();
}
EXPORT_SYMBOL(unregister_cpu_notifier);

void __unregister_cpu_notifier(struct notifier_block *nb)
{
 raw_notifier_chain_unregister(&cpu_chain, nb);
}
EXPORT_SYMBOL(__unregister_cpu_notifier);
void clear_tasks_mm_cpumask(int cpu)
{
 struct task_struct *p;
 WARN_ON(cpu_online(cpu));
 rcu_read_lock();
 for_each_process(p) {
  struct task_struct *t;





  t = find_lock_task_mm(p);
  if (!t)
   continue;
  cpumask_clear_cpu(cpu, mm_cpumask(t->mm));
  task_unlock(t);
 }
 rcu_read_unlock();
}

static inline void check_for_tasks(int dead_cpu)
{
 struct task_struct *g, *p;

 read_lock(&tasklist_lock);
 for_each_process_thread(g, p) {
  if (!p->on_rq)
   continue;






  rmb();
  if (task_cpu(p) != dead_cpu)
   continue;

  pr_warn("Task %s (pid=%d) is on cpu %d (state=%ld, flags=%x)\n",
   p->comm, task_pid_nr(p), dead_cpu, p->state, p->flags);
 }
 read_unlock(&tasklist_lock);
}

static int notify_down_prepare(unsigned int cpu)
{
 int err, nr_calls = 0;

 err = __cpu_notify(CPU_DOWN_PREPARE, cpu, -1, &nr_calls);
 if (err) {
  nr_calls--;
  __cpu_notify(CPU_DOWN_FAILED, cpu, nr_calls, NULL);
  pr_warn("%s: attempt to take down CPU %u failed\n",
    __func__, cpu);
 }
 return err;
}

static int notify_dying(unsigned int cpu)
{
 cpu_notify(CPU_DYING, cpu);
 return 0;
}


static int take_cpu_down(void *_param)
{
 struct cpuhp_cpu_state *st = this_cpu_ptr(&cpuhp_state);
 enum cpuhp_state target = max((int)st->target, CPUHP_AP_OFFLINE);
 int err, cpu = smp_processor_id();


 err = __cpu_disable();
 if (err < 0)
  return err;


 for (; st->state > target; st->state--) {
  struct cpuhp_step *step = cpuhp_ap_states + st->state;

  cpuhp_invoke_callback(cpu, st->state, step->teardown);
 }

 tick_handover_do_timer();

 stop_machine_park(cpu);
 return 0;
}

static int takedown_cpu(unsigned int cpu)
{
 struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, cpu);
 int err;


 kthread_park(per_cpu_ptr(&cpuhp_state, cpu)->thread);
 smpboot_park_threads(cpu);





 irq_lock_sparse();




 err = stop_machine(take_cpu_down, NULL, cpumask_of(cpu));
 if (err) {

  irq_unlock_sparse();

  kthread_unpark(per_cpu_ptr(&cpuhp_state, cpu)->thread);
  return err;
 }
 BUG_ON(cpu_online(cpu));
 wait_for_completion(&st->done);
 BUG_ON(st->state != CPUHP_AP_IDLE_DEAD);


 irq_unlock_sparse();

 hotplug_cpu__broadcast_tick_pull(cpu);

 __cpu_die(cpu);

 tick_cleanup_dead_cpu(cpu);
 return 0;
}

static int notify_dead(unsigned int cpu)
{
 cpu_notify_nofail(CPU_DEAD, cpu);
 check_for_tasks(cpu);
 return 0;
}

static void cpuhp_complete_idle_dead(void *arg)
{
 struct cpuhp_cpu_state *st = arg;

 complete(&st->done);
}

void cpuhp_report_idle_dead(void)
{
 struct cpuhp_cpu_state *st = this_cpu_ptr(&cpuhp_state);

 BUG_ON(st->state != CPUHP_AP_OFFLINE);
 rcu_report_dead(smp_processor_id());
 st->state = CPUHP_AP_IDLE_DEAD;




 smp_call_function_single(cpumask_first(cpu_online_mask),
     cpuhp_complete_idle_dead, st, 0);
}




static int __ref _cpu_down(unsigned int cpu, int tasks_frozen,
      enum cpuhp_state target)
{
 struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, cpu);
 int prev_state, ret = 0;
 bool hasdied = false;

 if (num_online_cpus() == 1)
  return -EBUSY;

 if (!cpu_present(cpu))
  return -EINVAL;

 cpu_hotplug_begin();

 cpuhp_tasks_frozen = tasks_frozen;

 prev_state = st->state;
 st->target = target;




 if (st->state > CPUHP_TEARDOWN_CPU) {
  ret = cpuhp_kick_ap_work(cpu);




  if (ret)
   goto out;





  if (st->state > CPUHP_TEARDOWN_CPU)
   goto out;
 }




 ret = cpuhp_down_callbacks(cpu, st, cpuhp_bp_states, target);
 if (ret && st->state > CPUHP_TEARDOWN_CPU && st->state < prev_state) {
  st->target = prev_state;
  st->rollback = true;
  cpuhp_kick_ap_work(cpu);
 }

 hasdied = prev_state != st->state && st->state == CPUHP_OFFLINE;
out:
 cpu_hotplug_done();

 if (!ret && hasdied)
  cpu_notify_nofail(CPU_POST_DEAD, cpu);
 return ret;
}

static int do_cpu_down(unsigned int cpu, enum cpuhp_state target)
{
 int err;

 cpu_maps_update_begin();

 if (cpu_hotplug_disabled) {
  err = -EBUSY;
  goto out;
 }

 err = _cpu_down(cpu, 0, target);

out:
 cpu_maps_update_done();
 return err;
}
int cpu_down(unsigned int cpu)
{
 return do_cpu_down(cpu, CPUHP_OFFLINE);
}
EXPORT_SYMBOL(cpu_down);
void notify_cpu_starting(unsigned int cpu)
{
 struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, cpu);
 enum cpuhp_state target = min((int)st->target, CPUHP_AP_ONLINE);

 while (st->state < target) {
  struct cpuhp_step *step;

  st->state++;
  step = cpuhp_ap_states + st->state;
  cpuhp_invoke_callback(cpu, st->state, step->startup);
 }
}







void cpuhp_online_idle(enum cpuhp_state state)
{
 struct cpuhp_cpu_state *st = this_cpu_ptr(&cpuhp_state);
 unsigned int cpu = smp_processor_id();


 if (state != CPUHP_AP_ONLINE_IDLE)
  return;

 st->state = CPUHP_AP_ONLINE_IDLE;


 stop_machine_unpark(cpu);
 kthread_unpark(st->thread);


 if (st->target > CPUHP_AP_ONLINE_IDLE)
  __cpuhp_kick_ap_work(st);
 else
  complete(&st->done);
}


static int _cpu_up(unsigned int cpu, int tasks_frozen, enum cpuhp_state target)
{
 struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, cpu);
 struct task_struct *idle;
 int ret = 0;

 cpu_hotplug_begin();

 if (!cpu_present(cpu)) {
  ret = -EINVAL;
  goto out;
 }





 if (st->state >= target)
  goto out;

 if (st->state == CPUHP_OFFLINE) {

  idle = idle_thread_get(cpu);
  if (IS_ERR(idle)) {
   ret = PTR_ERR(idle);
   goto out;
  }
 }

 cpuhp_tasks_frozen = tasks_frozen;

 st->target = target;




 if (st->state > CPUHP_BRINGUP_CPU) {
  ret = cpuhp_kick_ap_work(cpu);




  if (ret)
   goto out;
 }






 target = min((int)target, CPUHP_BRINGUP_CPU);
 ret = cpuhp_up_callbacks(cpu, st, cpuhp_bp_states, target);
out:
 cpu_hotplug_done();
 return ret;
}

static int do_cpu_up(unsigned int cpu, enum cpuhp_state target)
{
 int err = 0;

 if (!cpu_possible(cpu)) {
  pr_err("can't online cpu %d because it is not configured as may-hotadd at boot time\n",
         cpu);
  pr_err("please check additional_cpus= boot parameter\n");
  return -EINVAL;
 }

 err = try_online_node(cpu_to_node(cpu));
 if (err)
  return err;

 cpu_maps_update_begin();

 if (cpu_hotplug_disabled) {
  err = -EBUSY;
  goto out;
 }

 err = _cpu_up(cpu, 0, target);
out:
 cpu_maps_update_done();
 return err;
}

int cpu_up(unsigned int cpu)
{
 return do_cpu_up(cpu, CPUHP_ONLINE);
}
EXPORT_SYMBOL_GPL(cpu_up);

static cpumask_var_t frozen_cpus;

int disable_nonboot_cpus(void)
{
 int cpu, first_cpu, error = 0;

 cpu_maps_update_begin();
 first_cpu = cpumask_first(cpu_online_mask);




 cpumask_clear(frozen_cpus);

 pr_info("Disabling non-boot CPUs ...\n");
 for_each_online_cpu(cpu) {
  if (cpu == first_cpu)
   continue;
  trace_suspend_resume(TPS("CPU_OFF"), cpu, true);
  error = _cpu_down(cpu, 1, CPUHP_OFFLINE);
  trace_suspend_resume(TPS("CPU_OFF"), cpu, false);
  if (!error)
   cpumask_set_cpu(cpu, frozen_cpus);
  else {
   pr_err("Error taking CPU%d down: %d\n", cpu, error);
   break;
  }
 }

 if (!error)
  BUG_ON(num_online_cpus() > 1);
 else
  pr_err("Non-boot CPUs are not disabled\n");






 cpu_hotplug_disabled++;

 cpu_maps_update_done();
 return error;
}

void __weak arch_enable_nonboot_cpus_begin(void)
{
}

void __weak arch_enable_nonboot_cpus_end(void)
{
}

void enable_nonboot_cpus(void)
{
 int cpu, error;


 cpu_maps_update_begin();
 WARN_ON(--cpu_hotplug_disabled < 0);
 if (cpumask_empty(frozen_cpus))
  goto out;

 pr_info("Enabling non-boot CPUs ...\n");

 arch_enable_nonboot_cpus_begin();

 for_each_cpu(cpu, frozen_cpus) {
  trace_suspend_resume(TPS("CPU_ON"), cpu, true);
  error = _cpu_up(cpu, 1, CPUHP_ONLINE);
  trace_suspend_resume(TPS("CPU_ON"), cpu, false);
  if (!error) {
   pr_info("CPU%d is up\n", cpu);
   continue;
  }
  pr_warn("Error taking CPU%d up: %d\n", cpu, error);
 }

 arch_enable_nonboot_cpus_end();

 cpumask_clear(frozen_cpus);
out:
 cpu_maps_update_done();
}

static int __init alloc_frozen_cpus(void)
{
 if (!alloc_cpumask_var(&frozen_cpus, GFP_KERNEL|__GFP_ZERO))
  return -ENOMEM;
 return 0;
}
core_initcall(alloc_frozen_cpus);
static int
cpu_hotplug_pm_callback(struct notifier_block *nb,
   unsigned long action, void *ptr)
{
 switch (action) {

 case PM_SUSPEND_PREPARE:
 case PM_HIBERNATION_PREPARE:
  cpu_hotplug_disable();
  break;

 case PM_POST_SUSPEND:
 case PM_POST_HIBERNATION:
  cpu_hotplug_enable();
  break;

 default:
  return NOTIFY_DONE;
 }

 return NOTIFY_OK;
}


static int __init cpu_hotplug_pm_sync_init(void)
{





 pm_notifier(cpu_hotplug_pm_callback, 0);
 return 0;
}
core_initcall(cpu_hotplug_pm_sync_init);




static struct cpuhp_step cpuhp_bp_states[] = {
 [CPUHP_OFFLINE] = {
  .name = "offline",
  .startup = NULL,
  .teardown = NULL,
 },
 [CPUHP_CREATE_THREADS]= {
  .name = "threads:create",
  .startup = smpboot_create_threads,
  .teardown = NULL,
  .cant_stop = true,
 },




 [CPUHP_NOTIFY_PREPARE] = {
  .name = "notify:prepare",
  .startup = notify_prepare,
  .teardown = notify_dead,
  .skip_onerr = true,
  .cant_stop = true,
 },

 [CPUHP_BRINGUP_CPU] = {
  .name = "cpu:bringup",
  .startup = bringup_cpu,
  .teardown = NULL,
  .cant_stop = true,
 },




 [CPUHP_TEARDOWN_CPU] = {
  .name = "cpu:teardown",
  .startup = NULL,
  .teardown = takedown_cpu,
  .cant_stop = true,
 },
};


static struct cpuhp_step cpuhp_ap_states[] = {

 [CPUHP_AP_IDLE_DEAD] = {
  .name = "idle:dead",
 },




 [CPUHP_AP_OFFLINE] = {
  .name = "ap:offline",
  .cant_stop = true,
 },

 [CPUHP_AP_SCHED_STARTING] = {
  .name = "sched:starting",
  .startup = sched_cpu_starting,
  .teardown = sched_cpu_dying,
 },





 [CPUHP_AP_NOTIFY_STARTING] = {
  .name = "notify:starting",
  .startup = notify_starting,
  .teardown = notify_dying,
  .skip_onerr = true,
  .cant_stop = true,
 },


 [CPUHP_AP_ONLINE] = {
  .name = "ap:online",
 },

 [CPUHP_AP_SMPBOOT_THREADS] = {
  .name = "smpboot:threads",
  .startup = smpboot_unpark_threads,
  .teardown = NULL,
 },




 [CPUHP_AP_NOTIFY_ONLINE] = {
  .name = "notify:online",
  .startup = notify_online,
  .teardown = notify_down_prepare,
  .skip_onerr = true,
 },





 [CPUHP_AP_ACTIVE] = {
  .name = "sched:active",
  .startup = sched_cpu_activate,
  .teardown = sched_cpu_deactivate,
 },


 [CPUHP_ONLINE] = {
  .name = "online",
  .startup = NULL,
  .teardown = NULL,
 },
};


static int cpuhp_cb_check(enum cpuhp_state state)
{
 if (state <= CPUHP_OFFLINE || state >= CPUHP_ONLINE)
  return -EINVAL;
 return 0;
}

static bool cpuhp_is_ap_state(enum cpuhp_state state)
{




 return state > CPUHP_BRINGUP_CPU && state != CPUHP_TEARDOWN_CPU;
}

static struct cpuhp_step *cpuhp_get_step(enum cpuhp_state state)
{
 struct cpuhp_step *sp;

 sp = cpuhp_is_ap_state(state) ? cpuhp_ap_states : cpuhp_bp_states;
 return sp + state;
}

static void cpuhp_store_callbacks(enum cpuhp_state state,
      const char *name,
      int (*startup)(unsigned int cpu),
      int (*teardown)(unsigned int cpu))
{

 struct cpuhp_step *sp;

 mutex_lock(&cpuhp_state_mutex);
 sp = cpuhp_get_step(state);
 sp->startup = startup;
 sp->teardown = teardown;
 sp->name = name;
 mutex_unlock(&cpuhp_state_mutex);
}

static void *cpuhp_get_teardown_cb(enum cpuhp_state state)
{
 return cpuhp_get_step(state)->teardown;
}





static int cpuhp_issue_call(int cpu, enum cpuhp_state state,
       int (*cb)(unsigned int), bool bringup)
{
 int ret;

 if (!cb)
  return 0;




 if (cpuhp_is_ap_state(state))
  ret = cpuhp_invoke_ap_callback(cpu, state, cb);
 else
  ret = cpuhp_invoke_callback(cpu, state, cb);
 ret = cpuhp_invoke_callback(cpu, state, cb);
 BUG_ON(ret && !bringup);
 return ret;
}






static void cpuhp_rollback_install(int failedcpu, enum cpuhp_state state,
       int (*teardown)(unsigned int cpu))
{
 int cpu;

 if (!teardown)
  return;


 for_each_present_cpu(cpu) {
  struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, cpu);
  int cpustate = st->state;

  if (cpu >= failedcpu)
   break;


  if (cpustate >= state)
   cpuhp_issue_call(cpu, state, teardown, false);
 }
}






static int cpuhp_reserve_state(enum cpuhp_state state)
{
 enum cpuhp_state i;

 mutex_lock(&cpuhp_state_mutex);
 for (i = CPUHP_AP_ONLINE_DYN; i <= CPUHP_AP_ONLINE_DYN_END; i++) {
  if (cpuhp_ap_states[i].name)
   continue;

  cpuhp_ap_states[i].name = "Reserved";
  mutex_unlock(&cpuhp_state_mutex);
  return i;
 }
 mutex_unlock(&cpuhp_state_mutex);
 WARN(1, "No more dynamic states available for CPU hotplug\n");
 return -ENOSPC;
}
int __cpuhp_setup_state(enum cpuhp_state state,
   const char *name, bool invoke,
   int (*startup)(unsigned int cpu),
   int (*teardown)(unsigned int cpu))
{
 int cpu, ret = 0;
 int dyn_state = 0;

 if (cpuhp_cb_check(state) || !name)
  return -EINVAL;

 get_online_cpus();


 if (state == CPUHP_AP_ONLINE_DYN) {
  dyn_state = 1;
  ret = cpuhp_reserve_state(state);
  if (ret < 0)
   goto out;
  state = ret;
 }

 cpuhp_store_callbacks(state, name, startup, teardown);

 if (!invoke || !startup)
  goto out;





 for_each_present_cpu(cpu) {
  struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, cpu);
  int cpustate = st->state;

  if (cpustate < state)
   continue;

  ret = cpuhp_issue_call(cpu, state, startup, true);
  if (ret) {
   cpuhp_rollback_install(cpu, state, teardown);
   cpuhp_store_callbacks(state, NULL, NULL, NULL);
   goto out;
  }
 }
out:
 put_online_cpus();
 if (!ret && dyn_state)
  return state;
 return ret;
}
EXPORT_SYMBOL(__cpuhp_setup_state);
void __cpuhp_remove_state(enum cpuhp_state state, bool invoke)
{
 int (*teardown)(unsigned int cpu) = cpuhp_get_teardown_cb(state);
 int cpu;

 BUG_ON(cpuhp_cb_check(state));

 get_online_cpus();

 if (!invoke || !teardown)
  goto remove;






 for_each_present_cpu(cpu) {
  struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, cpu);
  int cpustate = st->state;

  if (cpustate >= state)
   cpuhp_issue_call(cpu, state, teardown, false);
 }
remove:
 cpuhp_store_callbacks(state, NULL, NULL, NULL);
 put_online_cpus();
}
EXPORT_SYMBOL(__cpuhp_remove_state);

static ssize_t show_cpuhp_state(struct device *dev,
    struct device_attribute *attr, char *buf)
{
 struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, dev->id);

 return sprintf(buf, "%d\n", st->state);
}
static DEVICE_ATTR(state, 0444, show_cpuhp_state, NULL);

static ssize_t write_cpuhp_target(struct device *dev,
      struct device_attribute *attr,
      const char *buf, size_t count)
{
 struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, dev->id);
 struct cpuhp_step *sp;
 int target, ret;

 ret = kstrtoint(buf, 10, &target);
 if (ret)
  return ret;

 if (target < CPUHP_OFFLINE || target > CPUHP_ONLINE)
  return -EINVAL;
 if (target != CPUHP_OFFLINE && target != CPUHP_ONLINE)
  return -EINVAL;

 ret = lock_device_hotplug_sysfs();
 if (ret)
  return ret;

 mutex_lock(&cpuhp_state_mutex);
 sp = cpuhp_get_step(target);
 ret = !sp->name || sp->cant_stop ? -EINVAL : 0;
 mutex_unlock(&cpuhp_state_mutex);
 if (ret)
  return ret;

 if (st->state < target)
  ret = do_cpu_up(dev->id, target);
 else
  ret = do_cpu_down(dev->id, target);

 unlock_device_hotplug();
 return ret ? ret : count;
}

static ssize_t show_cpuhp_target(struct device *dev,
     struct device_attribute *attr, char *buf)
{
 struct cpuhp_cpu_state *st = per_cpu_ptr(&cpuhp_state, dev->id);

 return sprintf(buf, "%d\n", st->target);
}
static DEVICE_ATTR(target, 0644, show_cpuhp_target, write_cpuhp_target);

static struct attribute *cpuhp_cpu_attrs[] = {
 &dev_attr_state.attr,
 &dev_attr_target.attr,
 NULL
};

static struct attribute_group cpuhp_cpu_attr_group = {
 .attrs = cpuhp_cpu_attrs,
 .name = "hotplug",
 NULL
};

static ssize_t show_cpuhp_states(struct device *dev,
     struct device_attribute *attr, char *buf)
{
 ssize_t cur, res = 0;
 int i;

 mutex_lock(&cpuhp_state_mutex);
 for (i = CPUHP_OFFLINE; i <= CPUHP_ONLINE; i++) {
  struct cpuhp_step *sp = cpuhp_get_step(i);

  if (sp->name) {
   cur = sprintf(buf, "%3d: %s\n", i, sp->name);
   buf += cur;
   res += cur;
  }
 }
 mutex_unlock(&cpuhp_state_mutex);
 return res;
}
static DEVICE_ATTR(states, 0444, show_cpuhp_states, NULL);

static struct attribute *cpuhp_cpu_root_attrs[] = {
 &dev_attr_states.attr,
 NULL
};

static struct attribute_group cpuhp_cpu_root_attr_group = {
 .attrs = cpuhp_cpu_root_attrs,
 .name = "hotplug",
 NULL
};

static int __init cpuhp_sysfs_init(void)
{
 int cpu, ret;

 ret = sysfs_create_group(&cpu_subsys.dev_root->kobj,
     &cpuhp_cpu_root_attr_group);
 if (ret)
  return ret;

 for_each_possible_cpu(cpu) {
  struct device *dev = get_cpu_device(cpu);

  if (!dev)
   continue;
  ret = sysfs_create_group(&dev->kobj, &cpuhp_cpu_attr_group);
  if (ret)
   return ret;
 }
 return 0;
}
device_initcall(cpuhp_sysfs_init);

const unsigned long cpu_bit_bitmap[BITS_PER_LONG+1][BITS_TO_LONGS(NR_CPUS)] = {

 MASK_DECLARE_8(0), MASK_DECLARE_8(8),
 MASK_DECLARE_8(16), MASK_DECLARE_8(24),
 MASK_DECLARE_8(32), MASK_DECLARE_8(40),
 MASK_DECLARE_8(48), MASK_DECLARE_8(56),
};
EXPORT_SYMBOL_GPL(cpu_bit_bitmap);

const DECLARE_BITMAP(cpu_all_bits, NR_CPUS) = CPU_BITS_ALL;
EXPORT_SYMBOL(cpu_all_bits);

struct cpumask __cpu_possible_mask __read_mostly
 = {CPU_BITS_ALL};
struct cpumask __cpu_possible_mask __read_mostly;
EXPORT_SYMBOL(__cpu_possible_mask);

struct cpumask __cpu_online_mask __read_mostly;
EXPORT_SYMBOL(__cpu_online_mask);

struct cpumask __cpu_present_mask __read_mostly;
EXPORT_SYMBOL(__cpu_present_mask);

struct cpumask __cpu_active_mask __read_mostly;
EXPORT_SYMBOL(__cpu_active_mask);

void init_cpu_present(const struct cpumask *src)
{
 cpumask_copy(&__cpu_present_mask, src);
}

void init_cpu_possible(const struct cpumask *src)
{
 cpumask_copy(&__cpu_possible_mask, src);
}

void init_cpu_online(const struct cpumask *src)
{
 cpumask_copy(&__cpu_online_mask, src);
}




void __init boot_cpu_init(void)
{
 int cpu = smp_processor_id();


 set_cpu_online(cpu, true);
 set_cpu_active(cpu, true);
 set_cpu_present(cpu, true);
 set_cpu_possible(cpu, true);
}




void __init boot_cpu_state_init(void)
{
 per_cpu_ptr(&cpuhp_state, smp_processor_id())->state = CPUHP_ONLINE;
}


static bool migrate_one_irq(struct irq_desc *desc)
{
 struct irq_data *d = irq_desc_get_irq_data(desc);
 const struct cpumask *affinity = d->common->affinity;
 struct irq_chip *c;
 bool ret = false;





 if (irqd_is_per_cpu(d) ||
     !cpumask_test_cpu(smp_processor_id(), affinity))
  return false;

 if (cpumask_any_and(affinity, cpu_online_mask) >= nr_cpu_ids) {
  affinity = cpu_online_mask;
  ret = true;
 }

 c = irq_data_get_irq_chip(d);
 if (!c->irq_set_affinity) {
  pr_debug("IRQ%u: unable to set affinity\n", d->irq);
 } else {
  int r = irq_do_set_affinity(d, affinity, false);
  if (r)
   pr_warn_ratelimited("IRQ%u: set affinity failed(%d).\n",
         d->irq, r);
 }

 return ret;
}
void irq_migrate_all_off_this_cpu(void)
{
 unsigned int irq;
 struct irq_desc *desc;
 unsigned long flags;

 local_irq_save(flags);

 for_each_active_irq(irq) {
  bool affinity_broken;

  desc = irq_to_desc(irq);
  raw_spin_lock(&desc->lock);
  affinity_broken = migrate_one_irq(desc);
  raw_spin_unlock(&desc->lock);

  if (affinity_broken)
   pr_warn_ratelimited("IRQ%u no longer affine to CPU%u\n",
         irq, smp_processor_id());
 }

 local_irq_restore(flags);
}

static DEFINE_RWLOCK(cpu_pm_notifier_lock);
static RAW_NOTIFIER_HEAD(cpu_pm_notifier_chain);

static int cpu_pm_notify(enum cpu_pm_event event, int nr_to_call, int *nr_calls)
{
 int ret;

 ret = __raw_notifier_call_chain(&cpu_pm_notifier_chain, event, NULL,
  nr_to_call, nr_calls);

 return notifier_to_errno(ret);
}
int cpu_pm_register_notifier(struct notifier_block *nb)
{
 unsigned long flags;
 int ret;

 write_lock_irqsave(&cpu_pm_notifier_lock, flags);
 ret = raw_notifier_chain_register(&cpu_pm_notifier_chain, nb);
 write_unlock_irqrestore(&cpu_pm_notifier_lock, flags);

 return ret;
}
EXPORT_SYMBOL_GPL(cpu_pm_register_notifier);
int cpu_pm_unregister_notifier(struct notifier_block *nb)
{
 unsigned long flags;
 int ret;

 write_lock_irqsave(&cpu_pm_notifier_lock, flags);
 ret = raw_notifier_chain_unregister(&cpu_pm_notifier_chain, nb);
 write_unlock_irqrestore(&cpu_pm_notifier_lock, flags);

 return ret;
}
EXPORT_SYMBOL_GPL(cpu_pm_unregister_notifier);
int cpu_pm_enter(void)
{
 int nr_calls;
 int ret = 0;

 read_lock(&cpu_pm_notifier_lock);
 ret = cpu_pm_notify(CPU_PM_ENTER, -1, &nr_calls);
 if (ret)




  cpu_pm_notify(CPU_PM_ENTER_FAILED, nr_calls - 1, NULL);
 read_unlock(&cpu_pm_notifier_lock);

 return ret;
}
EXPORT_SYMBOL_GPL(cpu_pm_enter);
int cpu_pm_exit(void)
{
 int ret;

 read_lock(&cpu_pm_notifier_lock);
 ret = cpu_pm_notify(CPU_PM_EXIT, -1, NULL);
 read_unlock(&cpu_pm_notifier_lock);

 return ret;
}
EXPORT_SYMBOL_GPL(cpu_pm_exit);
int cpu_cluster_pm_enter(void)
{
 int nr_calls;
 int ret = 0;

 read_lock(&cpu_pm_notifier_lock);
 ret = cpu_pm_notify(CPU_CLUSTER_PM_ENTER, -1, &nr_calls);
 if (ret)




  cpu_pm_notify(CPU_CLUSTER_PM_ENTER_FAILED, nr_calls - 1, NULL);
 read_unlock(&cpu_pm_notifier_lock);

 return ret;
}
EXPORT_SYMBOL_GPL(cpu_cluster_pm_enter);
int cpu_cluster_pm_exit(void)
{
 int ret;

 read_lock(&cpu_pm_notifier_lock);
 ret = cpu_pm_notify(CPU_CLUSTER_PM_EXIT, -1, NULL);
 read_unlock(&cpu_pm_notifier_lock);

 return ret;
}
EXPORT_SYMBOL_GPL(cpu_cluster_pm_exit);

static int cpu_pm_suspend(void)
{
 int ret;

 ret = cpu_pm_enter();
 if (ret)
  return ret;

 ret = cpu_cluster_pm_enter();
 return ret;
}

static void cpu_pm_resume(void)
{
 cpu_cluster_pm_exit();
 cpu_pm_exit();
}

static struct syscore_ops cpu_pm_syscore_ops = {
 .suspend = cpu_pm_suspend,
 .resume = cpu_pm_resume,
};

static int cpu_pm_init(void)
{
 register_syscore_ops(&cpu_pm_syscore_ops);
 return 0;
}
core_initcall(cpu_pm_init);


DEFINE_STATIC_KEY_FALSE(cpusets_enabled_key);



struct fmeter {
 int cnt;
 int val;
 time64_t time;
 spinlock_t lock;
};

struct cpuset {
 struct cgroup_subsys_state css;

 unsigned long flags;
 cpumask_var_t cpus_allowed;
 nodemask_t mems_allowed;


 cpumask_var_t effective_cpus;
 nodemask_t effective_mems;
 nodemask_t old_mems_allowed;

 struct fmeter fmeter;





 int attach_in_progress;


 int pn;


 int relax_domain_level;
};

static inline struct cpuset *css_cs(struct cgroup_subsys_state *css)
{
 return css ? container_of(css, struct cpuset, css) : NULL;
}


static inline struct cpuset *task_cs(struct task_struct *task)
{
 return css_cs(task_css(task, cpuset_cgrp_id));
}

static inline struct cpuset *parent_cs(struct cpuset *cs)
{
 return css_cs(cs->css.parent);
}

static inline bool task_has_mempolicy(struct task_struct *task)
{
 return task->mempolicy;
}
static inline bool task_has_mempolicy(struct task_struct *task)
{
 return false;
}



typedef enum {
 CS_ONLINE,
 CS_CPU_EXCLUSIVE,
 CS_MEM_EXCLUSIVE,
 CS_MEM_HARDWALL,
 CS_MEMORY_MIGRATE,
 CS_SCHED_LOAD_BALANCE,
 CS_SPREAD_PAGE,
 CS_SPREAD_SLAB,
} cpuset_flagbits_t;


static inline bool is_cpuset_online(const struct cpuset *cs)
{
 return test_bit(CS_ONLINE, &cs->flags);
}

static inline int is_cpu_exclusive(const struct cpuset *cs)
{
 return test_bit(CS_CPU_EXCLUSIVE, &cs->flags);
}

static inline int is_mem_exclusive(const struct cpuset *cs)
{
 return test_bit(CS_MEM_EXCLUSIVE, &cs->flags);
}

static inline int is_mem_hardwall(const struct cpuset *cs)
{
 return test_bit(CS_MEM_HARDWALL, &cs->flags);
}

static inline int is_sched_load_balance(const struct cpuset *cs)
{
 return test_bit(CS_SCHED_LOAD_BALANCE, &cs->flags);
}

static inline int is_memory_migrate(const struct cpuset *cs)
{
 return test_bit(CS_MEMORY_MIGRATE, &cs->flags);
}

static inline int is_spread_page(const struct cpuset *cs)
{
 return test_bit(CS_SPREAD_PAGE, &cs->flags);
}

static inline int is_spread_slab(const struct cpuset *cs)
{
 return test_bit(CS_SPREAD_SLAB, &cs->flags);
}

static struct cpuset top_cpuset = {
 .flags = ((1 << CS_ONLINE) | (1 << CS_CPU_EXCLUSIVE) |
    (1 << CS_MEM_EXCLUSIVE)),
};
 css_for_each_child((pos_css), &(parent_cs)->css) \
  if (is_cpuset_online(((child_cs) = css_cs((pos_css)))))
 css_for_each_descendant_pre((pos_css), &(root_cs)->css) \
  if (is_cpuset_online(((des_cs) = css_cs((pos_css)))))
static DEFINE_MUTEX(cpuset_mutex);
static DEFINE_SPINLOCK(callback_lock);

static struct workqueue_struct *cpuset_migrate_mm_wq;




static void cpuset_hotplug_workfn(struct work_struct *work);
static DECLARE_WORK(cpuset_hotplug_work, cpuset_hotplug_workfn);

static DECLARE_WAIT_QUEUE_HEAD(cpuset_attach_wq);






static struct dentry *cpuset_mount(struct file_system_type *fs_type,
    int flags, const char *unused_dev_name, void *data)
{
 struct file_system_type *cgroup_fs = get_fs_type("cgroup");
 struct dentry *ret = ERR_PTR(-ENODEV);
 if (cgroup_fs) {
  char mountopts[] =
   "cpuset,noprefix,"
   "release_agent=/sbin/cpuset_release_agent";
  ret = cgroup_fs->mount(cgroup_fs, flags,
        unused_dev_name, mountopts);
  put_filesystem(cgroup_fs);
 }
 return ret;
}

static struct file_system_type cpuset_fs_type = {
 .name = "cpuset",
 .mount = cpuset_mount,
};
static void guarantee_online_cpus(struct cpuset *cs, struct cpumask *pmask)
{
 while (!cpumask_intersects(cs->effective_cpus, cpu_online_mask))
  cs = parent_cs(cs);
 cpumask_and(pmask, cs->effective_cpus, cpu_online_mask);
}
static void guarantee_online_mems(struct cpuset *cs, nodemask_t *pmask)
{
 while (!nodes_intersects(cs->effective_mems, node_states[N_MEMORY]))
  cs = parent_cs(cs);
 nodes_and(*pmask, cs->effective_mems, node_states[N_MEMORY]);
}






static void cpuset_update_task_spread_flag(struct cpuset *cs,
     struct task_struct *tsk)
{
 if (is_spread_page(cs))
  task_set_spread_page(tsk);
 else
  task_clear_spread_page(tsk);

 if (is_spread_slab(cs))
  task_set_spread_slab(tsk);
 else
  task_clear_spread_slab(tsk);
}
static int is_cpuset_subset(const struct cpuset *p, const struct cpuset *q)
{
 return cpumask_subset(p->cpus_allowed, q->cpus_allowed) &&
  nodes_subset(p->mems_allowed, q->mems_allowed) &&
  is_cpu_exclusive(p) <= is_cpu_exclusive(q) &&
  is_mem_exclusive(p) <= is_mem_exclusive(q);
}





static struct cpuset *alloc_trial_cpuset(struct cpuset *cs)
{
 struct cpuset *trial;

 trial = kmemdup(cs, sizeof(*cs), GFP_KERNEL);
 if (!trial)
  return NULL;

 if (!alloc_cpumask_var(&trial->cpus_allowed, GFP_KERNEL))
  goto free_cs;
 if (!alloc_cpumask_var(&trial->effective_cpus, GFP_KERNEL))
  goto free_cpus;

 cpumask_copy(trial->cpus_allowed, cs->cpus_allowed);
 cpumask_copy(trial->effective_cpus, cs->effective_cpus);
 return trial;

free_cpus:
 free_cpumask_var(trial->cpus_allowed);
free_cs:
 kfree(trial);
 return NULL;
}





static void free_trial_cpuset(struct cpuset *trial)
{
 free_cpumask_var(trial->effective_cpus);
 free_cpumask_var(trial->cpus_allowed);
 kfree(trial);
}
static int validate_change(struct cpuset *cur, struct cpuset *trial)
{
 struct cgroup_subsys_state *css;
 struct cpuset *c, *par;
 int ret;

 rcu_read_lock();


 ret = -EBUSY;
 cpuset_for_each_child(c, css, cur)
  if (!is_cpuset_subset(c, trial))
   goto out;


 ret = 0;
 if (cur == &top_cpuset)
  goto out;

 par = parent_cs(cur);


 ret = -EACCES;
 if (!cgroup_subsys_on_dfl(cpuset_cgrp_subsys) &&
     !is_cpuset_subset(trial, par))
  goto out;





 ret = -EINVAL;
 cpuset_for_each_child(c, css, par) {
  if ((is_cpu_exclusive(trial) || is_cpu_exclusive(c)) &&
      c != cur &&
      cpumask_intersects(trial->cpus_allowed, c->cpus_allowed))
   goto out;
  if ((is_mem_exclusive(trial) || is_mem_exclusive(c)) &&
      c != cur &&
      nodes_intersects(trial->mems_allowed, c->mems_allowed))
   goto out;
 }





 ret = -ENOSPC;
 if ((cgroup_is_populated(cur->css.cgroup) || cur->attach_in_progress)) {
  if (!cpumask_empty(cur->cpus_allowed) &&
      cpumask_empty(trial->cpus_allowed))
   goto out;
  if (!nodes_empty(cur->mems_allowed) &&
      nodes_empty(trial->mems_allowed))
   goto out;
 }





 ret = -EBUSY;
 if (is_cpu_exclusive(cur) &&
     !cpuset_cpumask_can_shrink(cur->cpus_allowed,
           trial->cpus_allowed))
  goto out;

 ret = 0;
out:
 rcu_read_unlock();
 return ret;
}





static int cpusets_overlap(struct cpuset *a, struct cpuset *b)
{
 return cpumask_intersects(a->effective_cpus, b->effective_cpus);
}

static void
update_domain_attr(struct sched_domain_attr *dattr, struct cpuset *c)
{
 if (dattr->relax_domain_level < c->relax_domain_level)
  dattr->relax_domain_level = c->relax_domain_level;
 return;
}

static void update_domain_attr_tree(struct sched_domain_attr *dattr,
        struct cpuset *root_cs)
{
 struct cpuset *cp;
 struct cgroup_subsys_state *pos_css;

 rcu_read_lock();
 cpuset_for_each_descendant_pre(cp, pos_css, root_cs) {

  if (cpumask_empty(cp->cpus_allowed)) {
   pos_css = css_rightmost_descendant(pos_css);
   continue;
  }

  if (is_sched_load_balance(cp))
   update_domain_attr(dattr, cp);
 }
 rcu_read_unlock();
}
static int generate_sched_domains(cpumask_var_t **domains,
   struct sched_domain_attr **attributes)
{
 struct cpuset *cp;
 struct cpuset **csa;
 int csn;
 int i, j, k;
 cpumask_var_t *doms;
 cpumask_var_t non_isolated_cpus;
 struct sched_domain_attr *dattr;
 int ndoms = 0;
 int nslot;
 struct cgroup_subsys_state *pos_css;

 doms = NULL;
 dattr = NULL;
 csa = NULL;

 if (!alloc_cpumask_var(&non_isolated_cpus, GFP_KERNEL))
  goto done;
 cpumask_andnot(non_isolated_cpus, cpu_possible_mask, cpu_isolated_map);


 if (is_sched_load_balance(&top_cpuset)) {
  ndoms = 1;
  doms = alloc_sched_domains(ndoms);
  if (!doms)
   goto done;

  dattr = kmalloc(sizeof(struct sched_domain_attr), GFP_KERNEL);
  if (dattr) {
   *dattr = SD_ATTR_INIT;
   update_domain_attr_tree(dattr, &top_cpuset);
  }
  cpumask_and(doms[0], top_cpuset.effective_cpus,
         non_isolated_cpus);

  goto done;
 }

 csa = kmalloc(nr_cpusets() * sizeof(cp), GFP_KERNEL);
 if (!csa)
  goto done;
 csn = 0;

 rcu_read_lock();
 cpuset_for_each_descendant_pre(cp, pos_css, &top_cpuset) {
  if (cp == &top_cpuset)
   continue;
  if (!cpumask_empty(cp->cpus_allowed) &&
      !(is_sched_load_balance(cp) &&
        cpumask_intersects(cp->cpus_allowed, non_isolated_cpus)))
   continue;

  if (is_sched_load_balance(cp))
   csa[csn++] = cp;


  pos_css = css_rightmost_descendant(pos_css);
 }
 rcu_read_unlock();

 for (i = 0; i < csn; i++)
  csa[i]->pn = i;
 ndoms = csn;

restart:

 for (i = 0; i < csn; i++) {
  struct cpuset *a = csa[i];
  int apn = a->pn;

  for (j = 0; j < csn; j++) {
   struct cpuset *b = csa[j];
   int bpn = b->pn;

   if (apn != bpn && cpusets_overlap(a, b)) {
    for (k = 0; k < csn; k++) {
     struct cpuset *c = csa[k];

     if (c->pn == bpn)
      c->pn = apn;
    }
    ndoms--;
    goto restart;
   }
  }
 }





 doms = alloc_sched_domains(ndoms);
 if (!doms)
  goto done;





 dattr = kmalloc(ndoms * sizeof(struct sched_domain_attr), GFP_KERNEL);

 for (nslot = 0, i = 0; i < csn; i++) {
  struct cpuset *a = csa[i];
  struct cpumask *dp;
  int apn = a->pn;

  if (apn < 0) {

   continue;
  }

  dp = doms[nslot];

  if (nslot == ndoms) {
   static int warnings = 10;
   if (warnings) {
    pr_warn("rebuild_sched_domains confused: nslot %d, ndoms %d, csn %d, i %d, apn %d\n",
     nslot, ndoms, csn, i, apn);
    warnings--;
   }
   continue;
  }

  cpumask_clear(dp);
  if (dattr)
   *(dattr + nslot) = SD_ATTR_INIT;
  for (j = i; j < csn; j++) {
   struct cpuset *b = csa[j];

   if (apn == b->pn) {
    cpumask_or(dp, dp, b->effective_cpus);
    cpumask_and(dp, dp, non_isolated_cpus);
    if (dattr)
     update_domain_attr_tree(dattr + nslot, b);


    b->pn = -1;
   }
  }
  nslot++;
 }
 BUG_ON(nslot != ndoms);

done:
 free_cpumask_var(non_isolated_cpus);
 kfree(csa);





 if (doms == NULL)
  ndoms = 1;

 *domains = doms;
 *attributes = dattr;
 return ndoms;
}
static void rebuild_sched_domains_locked(void)
{
 struct sched_domain_attr *attr;
 cpumask_var_t *doms;
 int ndoms;

 lockdep_assert_held(&cpuset_mutex);
 get_online_cpus();






 if (!cpumask_equal(top_cpuset.effective_cpus, cpu_active_mask))
  goto out;


 ndoms = generate_sched_domains(&doms, &attr);


 partition_sched_domains(ndoms, doms, attr);
out:
 put_online_cpus();
}
static void rebuild_sched_domains_locked(void)
{
}

void rebuild_sched_domains(void)
{
 mutex_lock(&cpuset_mutex);
 rebuild_sched_domains_locked();
 mutex_unlock(&cpuset_mutex);
}
static void update_tasks_cpumask(struct cpuset *cs)
{
 struct css_task_iter it;
 struct task_struct *task;

 css_task_iter_start(&cs->css, &it);
 while ((task = css_task_iter_next(&it)))
  set_cpus_allowed_ptr(task, cs->effective_cpus);
 css_task_iter_end(&it);
}
static void update_cpumasks_hier(struct cpuset *cs, struct cpumask *new_cpus)
{
 struct cpuset *cp;
 struct cgroup_subsys_state *pos_css;
 bool need_rebuild_sched_domains = false;

 rcu_read_lock();
 cpuset_for_each_descendant_pre(cp, pos_css, cs) {
  struct cpuset *parent = parent_cs(cp);

  cpumask_and(new_cpus, cp->cpus_allowed, parent->effective_cpus);





  if (cgroup_subsys_on_dfl(cpuset_cgrp_subsys) &&
      cpumask_empty(new_cpus))
   cpumask_copy(new_cpus, parent->effective_cpus);


  if (cpumask_equal(new_cpus, cp->effective_cpus)) {
   pos_css = css_rightmost_descendant(pos_css);
   continue;
  }

  if (!css_tryget_online(&cp->css))
   continue;
  rcu_read_unlock();

  spin_lock_irq(&callback_lock);
  cpumask_copy(cp->effective_cpus, new_cpus);
  spin_unlock_irq(&callback_lock);

  WARN_ON(!cgroup_subsys_on_dfl(cpuset_cgrp_subsys) &&
   !cpumask_equal(cp->cpus_allowed, cp->effective_cpus));

  update_tasks_cpumask(cp);





  if (!cpumask_empty(cp->cpus_allowed) &&
      is_sched_load_balance(cp))
   need_rebuild_sched_domains = true;

  rcu_read_lock();
  css_put(&cp->css);
 }
 rcu_read_unlock();

 if (need_rebuild_sched_domains)
  rebuild_sched_domains_locked();
}







static int update_cpumask(struct cpuset *cs, struct cpuset *trialcs,
     const char *buf)
{
 int retval;


 if (cs == &top_cpuset)
  return -EACCES;







 if (!*buf) {
  cpumask_clear(trialcs->cpus_allowed);
 } else {
  retval = cpulist_parse(buf, trialcs->cpus_allowed);
  if (retval < 0)
   return retval;

  if (!cpumask_subset(trialcs->cpus_allowed,
        top_cpuset.cpus_allowed))
   return -EINVAL;
 }


 if (cpumask_equal(cs->cpus_allowed, trialcs->cpus_allowed))
  return 0;

 retval = validate_change(cs, trialcs);
 if (retval < 0)
  return retval;

 spin_lock_irq(&callback_lock);
 cpumask_copy(cs->cpus_allowed, trialcs->cpus_allowed);
 spin_unlock_irq(&callback_lock);


 update_cpumasks_hier(cs, trialcs->cpus_allowed);
 return 0;
}
struct cpuset_migrate_mm_work {
 struct work_struct work;
 struct mm_struct *mm;
 nodemask_t from;
 nodemask_t to;
};

static void cpuset_migrate_mm_workfn(struct work_struct *work)
{
 struct cpuset_migrate_mm_work *mwork =
  container_of(work, struct cpuset_migrate_mm_work, work);


 do_migrate_pages(mwork->mm, &mwork->from, &mwork->to, MPOL_MF_MOVE_ALL);
 mmput(mwork->mm);
 kfree(mwork);
}

static void cpuset_migrate_mm(struct mm_struct *mm, const nodemask_t *from,
       const nodemask_t *to)
{
 struct cpuset_migrate_mm_work *mwork;

 mwork = kzalloc(sizeof(*mwork), GFP_KERNEL);
 if (mwork) {
  mwork->mm = mm;
  mwork->from = *from;
  mwork->to = *to;
  INIT_WORK(&mwork->work, cpuset_migrate_mm_workfn);
  queue_work(cpuset_migrate_mm_wq, &mwork->work);
 } else {
  mmput(mm);
 }
}

static void cpuset_post_attach(void)
{
 flush_workqueue(cpuset_migrate_mm_wq);
}
static void cpuset_change_task_nodemask(struct task_struct *tsk,
     nodemask_t *newmems)
{
 bool need_loop;





 if (unlikely(test_thread_flag(TIF_MEMDIE)))
  return;
 if (current->flags & PF_EXITING)
  return;

 task_lock(tsk);






 need_loop = task_has_mempolicy(tsk) ||
   !nodes_intersects(*newmems, tsk->mems_allowed);

 if (need_loop) {
  local_irq_disable();
  write_seqcount_begin(&tsk->mems_allowed_seq);
 }

 nodes_or(tsk->mems_allowed, tsk->mems_allowed, *newmems);
 mpol_rebind_task(tsk, newmems, MPOL_REBIND_STEP1);

 mpol_rebind_task(tsk, newmems, MPOL_REBIND_STEP2);
 tsk->mems_allowed = *newmems;

 if (need_loop) {
  write_seqcount_end(&tsk->mems_allowed_seq);
  local_irq_enable();
 }

 task_unlock(tsk);
}

static void *cpuset_being_rebound;
static void update_tasks_nodemask(struct cpuset *cs)
{
 static nodemask_t newmems;
 struct css_task_iter it;
 struct task_struct *task;

 cpuset_being_rebound = cs;

 guarantee_online_mems(cs, &newmems);
 css_task_iter_start(&cs->css, &it);
 while ((task = css_task_iter_next(&it))) {
  struct mm_struct *mm;
  bool migrate;

  cpuset_change_task_nodemask(task, &newmems);

  mm = get_task_mm(task);
  if (!mm)
   continue;

  migrate = is_memory_migrate(cs);

  mpol_rebind_mm(mm, &cs->mems_allowed);
  if (migrate)
   cpuset_migrate_mm(mm, &cs->old_mems_allowed, &newmems);
  else
   mmput(mm);
 }
 css_task_iter_end(&it);





 cs->old_mems_allowed = newmems;


 cpuset_being_rebound = NULL;
}
static void update_nodemasks_hier(struct cpuset *cs, nodemask_t *new_mems)
{
 struct cpuset *cp;
 struct cgroup_subsys_state *pos_css;

 rcu_read_lock();
 cpuset_for_each_descendant_pre(cp, pos_css, cs) {
  struct cpuset *parent = parent_cs(cp);

  nodes_and(*new_mems, cp->mems_allowed, parent->effective_mems);





  if (cgroup_subsys_on_dfl(cpuset_cgrp_subsys) &&
      nodes_empty(*new_mems))
   *new_mems = parent->effective_mems;


  if (nodes_equal(*new_mems, cp->effective_mems)) {
   pos_css = css_rightmost_descendant(pos_css);
   continue;
  }

  if (!css_tryget_online(&cp->css))
   continue;
  rcu_read_unlock();

  spin_lock_irq(&callback_lock);
  cp->effective_mems = *new_mems;
  spin_unlock_irq(&callback_lock);

  WARN_ON(!cgroup_subsys_on_dfl(cpuset_cgrp_subsys) &&
   !nodes_equal(cp->mems_allowed, cp->effective_mems));

  update_tasks_nodemask(cp);

  rcu_read_lock();
  css_put(&cp->css);
 }
 rcu_read_unlock();
}
static int update_nodemask(struct cpuset *cs, struct cpuset *trialcs,
      const char *buf)
{
 int retval;





 if (cs == &top_cpuset) {
  retval = -EACCES;
  goto done;
 }







 if (!*buf) {
  nodes_clear(trialcs->mems_allowed);
 } else {
  retval = nodelist_parse(buf, trialcs->mems_allowed);
  if (retval < 0)
   goto done;

  if (!nodes_subset(trialcs->mems_allowed,
      top_cpuset.mems_allowed)) {
   retval = -EINVAL;
   goto done;
  }
 }

 if (nodes_equal(cs->mems_allowed, trialcs->mems_allowed)) {
  retval = 0;
  goto done;
 }
 retval = validate_change(cs, trialcs);
 if (retval < 0)
  goto done;

 spin_lock_irq(&callback_lock);
 cs->mems_allowed = trialcs->mems_allowed;
 spin_unlock_irq(&callback_lock);


 update_nodemasks_hier(cs, &trialcs->mems_allowed);
done:
 return retval;
}

int current_cpuset_is_being_rebound(void)
{
 int ret;

 rcu_read_lock();
 ret = task_cs(current) == cpuset_being_rebound;
 rcu_read_unlock();

 return ret;
}

static int update_relax_domain_level(struct cpuset *cs, s64 val)
{
 if (val < -1 || val >= sched_domain_level_max)
  return -EINVAL;

 if (val != cs->relax_domain_level) {
  cs->relax_domain_level = val;
  if (!cpumask_empty(cs->cpus_allowed) &&
      is_sched_load_balance(cs))
   rebuild_sched_domains_locked();
 }

 return 0;
}
static void update_tasks_flags(struct cpuset *cs)
{
 struct css_task_iter it;
 struct task_struct *task;

 css_task_iter_start(&cs->css, &it);
 while ((task = css_task_iter_next(&it)))
  cpuset_update_task_spread_flag(cs, task);
 css_task_iter_end(&it);
}
static int update_flag(cpuset_flagbits_t bit, struct cpuset *cs,
         int turning_on)
{
 struct cpuset *trialcs;
 int balance_flag_changed;
 int spread_flag_changed;
 int err;

 trialcs = alloc_trial_cpuset(cs);
 if (!trialcs)
  return -ENOMEM;

 if (turning_on)
  set_bit(bit, &trialcs->flags);
 else
  clear_bit(bit, &trialcs->flags);

 err = validate_change(cs, trialcs);
 if (err < 0)
  goto out;

 balance_flag_changed = (is_sched_load_balance(cs) !=
    is_sched_load_balance(trialcs));

 spread_flag_changed = ((is_spread_slab(cs) != is_spread_slab(trialcs))
   || (is_spread_page(cs) != is_spread_page(trialcs)));

 spin_lock_irq(&callback_lock);
 cs->flags = trialcs->flags;
 spin_unlock_irq(&callback_lock);

 if (!cpumask_empty(trialcs->cpus_allowed) && balance_flag_changed)
  rebuild_sched_domains_locked();

 if (spread_flag_changed)
  update_tasks_flags(cs);
out:
 free_trial_cpuset(trialcs);
 return err;
}


static void fmeter_init(struct fmeter *fmp)
{
 fmp->cnt = 0;
 fmp->val = 0;
 fmp->time = 0;
 spin_lock_init(&fmp->lock);
}


static void fmeter_update(struct fmeter *fmp)
{
 time64_t now;
 u32 ticks;

 now = ktime_get_seconds();
 ticks = now - fmp->time;

 if (ticks == 0)
  return;

 ticks = min(FM_MAXTICKS, ticks);
 while (ticks-- > 0)
  fmp->val = (FM_COEF * fmp->val) / FM_SCALE;
 fmp->time = now;

 fmp->val += ((FM_SCALE - FM_COEF) * fmp->cnt) / FM_SCALE;
 fmp->cnt = 0;
}


static void fmeter_markevent(struct fmeter *fmp)
{
 spin_lock(&fmp->lock);
 fmeter_update(fmp);
 fmp->cnt = min(FM_MAXCNT, fmp->cnt + FM_SCALE);
 spin_unlock(&fmp->lock);
}


static int fmeter_getrate(struct fmeter *fmp)
{
 int val;

 spin_lock(&fmp->lock);
 fmeter_update(fmp);
 val = fmp->val;
 spin_unlock(&fmp->lock);
 return val;
}

static struct cpuset *cpuset_attach_old_cs;


static int cpuset_can_attach(struct cgroup_taskset *tset)
{
 struct cgroup_subsys_state *css;
 struct cpuset *cs;
 struct task_struct *task;
 int ret;


 cpuset_attach_old_cs = task_cs(cgroup_taskset_first(tset, &css));
 cs = css_cs(css);

 mutex_lock(&cpuset_mutex);


 ret = -ENOSPC;
 if (!cgroup_subsys_on_dfl(cpuset_cgrp_subsys) &&
     (cpumask_empty(cs->cpus_allowed) || nodes_empty(cs->mems_allowed)))
  goto out_unlock;

 cgroup_taskset_for_each(task, css, tset) {
  ret = task_can_attach(task, cs->cpus_allowed);
  if (ret)
   goto out_unlock;
  ret = security_task_setscheduler(task);
  if (ret)
   goto out_unlock;
 }





 cs->attach_in_progress++;
 ret = 0;
out_unlock:
 mutex_unlock(&cpuset_mutex);
 return ret;
}

static void cpuset_cancel_attach(struct cgroup_taskset *tset)
{
 struct cgroup_subsys_state *css;
 struct cpuset *cs;

 cgroup_taskset_first(tset, &css);
 cs = css_cs(css);

 mutex_lock(&cpuset_mutex);
 css_cs(css)->attach_in_progress--;
 mutex_unlock(&cpuset_mutex);
}






static cpumask_var_t cpus_attach;

static void cpuset_attach(struct cgroup_taskset *tset)
{

 static nodemask_t cpuset_attach_nodemask_to;
 struct task_struct *task;
 struct task_struct *leader;
 struct cgroup_subsys_state *css;
 struct cpuset *cs;
 struct cpuset *oldcs = cpuset_attach_old_cs;

 cgroup_taskset_first(tset, &css);
 cs = css_cs(css);

 mutex_lock(&cpuset_mutex);


 if (cs == &top_cpuset)
  cpumask_copy(cpus_attach, cpu_possible_mask);
 else
  guarantee_online_cpus(cs, cpus_attach);

 guarantee_online_mems(cs, &cpuset_attach_nodemask_to);

 cgroup_taskset_for_each(task, css, tset) {




  WARN_ON_ONCE(set_cpus_allowed_ptr(task, cpus_attach));

  cpuset_change_task_nodemask(task, &cpuset_attach_nodemask_to);
  cpuset_update_task_spread_flag(cs, task);
 }





 cpuset_attach_nodemask_to = cs->effective_mems;
 cgroup_taskset_for_each_leader(leader, css, tset) {
  struct mm_struct *mm = get_task_mm(leader);

  if (mm) {
   mpol_rebind_mm(mm, &cpuset_attach_nodemask_to);
   if (is_memory_migrate(cs))
    cpuset_migrate_mm(mm, &oldcs->old_mems_allowed,
        &cpuset_attach_nodemask_to);
   else
    mmput(mm);
  }
 }

 cs->old_mems_allowed = cpuset_attach_nodemask_to;

 cs->attach_in_progress--;
 if (!cs->attach_in_progress)
  wake_up(&cpuset_attach_wq);

 mutex_unlock(&cpuset_mutex);
}



typedef enum {
 FILE_MEMORY_MIGRATE,
 FILE_CPULIST,
 FILE_MEMLIST,
 FILE_EFFECTIVE_CPULIST,
 FILE_EFFECTIVE_MEMLIST,
 FILE_CPU_EXCLUSIVE,
 FILE_MEM_EXCLUSIVE,
 FILE_MEM_HARDWALL,
 FILE_SCHED_LOAD_BALANCE,
 FILE_SCHED_RELAX_DOMAIN_LEVEL,
 FILE_MEMORY_PRESSURE_ENABLED,
 FILE_MEMORY_PRESSURE,
 FILE_SPREAD_PAGE,
 FILE_SPREAD_SLAB,
} cpuset_filetype_t;

static int cpuset_write_u64(struct cgroup_subsys_state *css, struct cftype *cft,
       u64 val)
{
 struct cpuset *cs = css_cs(css);
 cpuset_filetype_t type = cft->private;
 int retval = 0;

 mutex_lock(&cpuset_mutex);
 if (!is_cpuset_online(cs)) {
  retval = -ENODEV;
  goto out_unlock;
 }

 switch (type) {
 case FILE_CPU_EXCLUSIVE:
  retval = update_flag(CS_CPU_EXCLUSIVE, cs, val);
  break;
 case FILE_MEM_EXCLUSIVE:
  retval = update_flag(CS_MEM_EXCLUSIVE, cs, val);
  break;
 case FILE_MEM_HARDWALL:
  retval = update_flag(CS_MEM_HARDWALL, cs, val);
  break;
 case FILE_SCHED_LOAD_BALANCE:
  retval = update_flag(CS_SCHED_LOAD_BALANCE, cs, val);
  break;
 case FILE_MEMORY_MIGRATE:
  retval = update_flag(CS_MEMORY_MIGRATE, cs, val);
  break;
 case FILE_MEMORY_PRESSURE_ENABLED:
  cpuset_memory_pressure_enabled = !!val;
  break;
 case FILE_SPREAD_PAGE:
  retval = update_flag(CS_SPREAD_PAGE, cs, val);
  break;
 case FILE_SPREAD_SLAB:
  retval = update_flag(CS_SPREAD_SLAB, cs, val);
  break;
 default:
  retval = -EINVAL;
  break;
 }
out_unlock:
 mutex_unlock(&cpuset_mutex);
 return retval;
}

static int cpuset_write_s64(struct cgroup_subsys_state *css, struct cftype *cft,
       s64 val)
{
 struct cpuset *cs = css_cs(css);
 cpuset_filetype_t type = cft->private;
 int retval = -ENODEV;

 mutex_lock(&cpuset_mutex);
 if (!is_cpuset_online(cs))
  goto out_unlock;

 switch (type) {
 case FILE_SCHED_RELAX_DOMAIN_LEVEL:
  retval = update_relax_domain_level(cs, val);
  break;
 default:
  retval = -EINVAL;
  break;
 }
out_unlock:
 mutex_unlock(&cpuset_mutex);
 return retval;
}




static ssize_t cpuset_write_resmask(struct kernfs_open_file *of,
        char *buf, size_t nbytes, loff_t off)
{
 struct cpuset *cs = css_cs(of_css(of));
 struct cpuset *trialcs;
 int retval = -ENODEV;

 buf = strstrip(buf);
 css_get(&cs->css);
 kernfs_break_active_protection(of->kn);
 flush_work(&cpuset_hotplug_work);

 mutex_lock(&cpuset_mutex);
 if (!is_cpuset_online(cs))
  goto out_unlock;

 trialcs = alloc_trial_cpuset(cs);
 if (!trialcs) {
  retval = -ENOMEM;
  goto out_unlock;
 }

 switch (of_cft(of)->private) {
 case FILE_CPULIST:
  retval = update_cpumask(cs, trialcs, buf);
  break;
 case FILE_MEMLIST:
  retval = update_nodemask(cs, trialcs, buf);
  break;
 default:
  retval = -EINVAL;
  break;
 }

 free_trial_cpuset(trialcs);
out_unlock:
 mutex_unlock(&cpuset_mutex);
 kernfs_unbreak_active_protection(of->kn);
 css_put(&cs->css);
 flush_workqueue(cpuset_migrate_mm_wq);
 return retval ?: nbytes;
}
static int cpuset_common_seq_show(struct seq_file *sf, void *v)
{
 struct cpuset *cs = css_cs(seq_css(sf));
 cpuset_filetype_t type = seq_cft(sf)->private;
 int ret = 0;

 spin_lock_irq(&callback_lock);

 switch (type) {
 case FILE_CPULIST:
  seq_printf(sf, "%*pbl\n", cpumask_pr_args(cs->cpus_allowed));
  break;
 case FILE_MEMLIST:
  seq_printf(sf, "%*pbl\n", nodemask_pr_args(&cs->mems_allowed));
  break;
 case FILE_EFFECTIVE_CPULIST:
  seq_printf(sf, "%*pbl\n", cpumask_pr_args(cs->effective_cpus));
  break;
 case FILE_EFFECTIVE_MEMLIST:
  seq_printf(sf, "%*pbl\n", nodemask_pr_args(&cs->effective_mems));
  break;
 default:
  ret = -EINVAL;
 }

 spin_unlock_irq(&callback_lock);
 return ret;
}

static u64 cpuset_read_u64(struct cgroup_subsys_state *css, struct cftype *cft)
{
 struct cpuset *cs = css_cs(css);
 cpuset_filetype_t type = cft->private;
 switch (type) {
 case FILE_CPU_EXCLUSIVE:
  return is_cpu_exclusive(cs);
 case FILE_MEM_EXCLUSIVE:
  return is_mem_exclusive(cs);
 case FILE_MEM_HARDWALL:
  return is_mem_hardwall(cs);
 case FILE_SCHED_LOAD_BALANCE:
  return is_sched_load_balance(cs);
 case FILE_MEMORY_MIGRATE:
  return is_memory_migrate(cs);
 case FILE_MEMORY_PRESSURE_ENABLED:
  return cpuset_memory_pressure_enabled;
 case FILE_MEMORY_PRESSURE:
  return fmeter_getrate(&cs->fmeter);
 case FILE_SPREAD_PAGE:
  return is_spread_page(cs);
 case FILE_SPREAD_SLAB:
  return is_spread_slab(cs);
 default:
  BUG();
 }


 return 0;
}

static s64 cpuset_read_s64(struct cgroup_subsys_state *css, struct cftype *cft)
{
 struct cpuset *cs = css_cs(css);
 cpuset_filetype_t type = cft->private;
 switch (type) {
 case FILE_SCHED_RELAX_DOMAIN_LEVEL:
  return cs->relax_domain_level;
 default:
  BUG();
 }


 return 0;
}






static struct cftype files[] = {
 {
  .name = "cpus",
  .seq_show = cpuset_common_seq_show,
  .write = cpuset_write_resmask,
  .max_write_len = (100U + 6 * NR_CPUS),
  .private = FILE_CPULIST,
 },

 {
  .name = "mems",
  .seq_show = cpuset_common_seq_show,
  .write = cpuset_write_resmask,
  .max_write_len = (100U + 6 * MAX_NUMNODES),
  .private = FILE_MEMLIST,
 },

 {
  .name = "effective_cpus",
  .seq_show = cpuset_common_seq_show,
  .private = FILE_EFFECTIVE_CPULIST,
 },

 {
  .name = "effective_mems",
  .seq_show = cpuset_common_seq_show,
  .private = FILE_EFFECTIVE_MEMLIST,
 },

 {
  .name = "cpu_exclusive",
  .read_u64 = cpuset_read_u64,
  .write_u64 = cpuset_write_u64,
  .private = FILE_CPU_EXCLUSIVE,
 },

 {
  .name = "mem_exclusive",
  .read_u64 = cpuset_read_u64,
  .write_u64 = cpuset_write_u64,
  .private = FILE_MEM_EXCLUSIVE,
 },

 {
  .name = "mem_hardwall",
  .read_u64 = cpuset_read_u64,
  .write_u64 = cpuset_write_u64,
  .private = FILE_MEM_HARDWALL,
 },

 {
  .name = "sched_load_balance",
  .read_u64 = cpuset_read_u64,
  .write_u64 = cpuset_write_u64,
  .private = FILE_SCHED_LOAD_BALANCE,
 },

 {
  .name = "sched_relax_domain_level",
  .read_s64 = cpuset_read_s64,
  .write_s64 = cpuset_write_s64,
  .private = FILE_SCHED_RELAX_DOMAIN_LEVEL,
 },

 {
  .name = "memory_migrate",
  .read_u64 = cpuset_read_u64,
  .write_u64 = cpuset_write_u64,
  .private = FILE_MEMORY_MIGRATE,
 },

 {
  .name = "memory_pressure",
  .read_u64 = cpuset_read_u64,
 },

 {
  .name = "memory_spread_page",
  .read_u64 = cpuset_read_u64,
  .write_u64 = cpuset_write_u64,
  .private = FILE_SPREAD_PAGE,
 },

 {
  .name = "memory_spread_slab",
  .read_u64 = cpuset_read_u64,
  .write_u64 = cpuset_write_u64,
  .private = FILE_SPREAD_SLAB,
 },

 {
  .name = "memory_pressure_enabled",
  .flags = CFTYPE_ONLY_ON_ROOT,
  .read_u64 = cpuset_read_u64,
  .write_u64 = cpuset_write_u64,
  .private = FILE_MEMORY_PRESSURE_ENABLED,
 },

 { }
};






static struct cgroup_subsys_state *
cpuset_css_alloc(struct cgroup_subsys_state *parent_css)
{
 struct cpuset *cs;

 if (!parent_css)
  return &top_cpuset.css;

 cs = kzalloc(sizeof(*cs), GFP_KERNEL);
 if (!cs)
  return ERR_PTR(-ENOMEM);
 if (!alloc_cpumask_var(&cs->cpus_allowed, GFP_KERNEL))
  goto free_cs;
 if (!alloc_cpumask_var(&cs->effective_cpus, GFP_KERNEL))
  goto free_cpus;

 set_bit(CS_SCHED_LOAD_BALANCE, &cs->flags);
 cpumask_clear(cs->cpus_allowed);
 nodes_clear(cs->mems_allowed);
 cpumask_clear(cs->effective_cpus);
 nodes_clear(cs->effective_mems);
 fmeter_init(&cs->fmeter);
 cs->relax_domain_level = -1;

 return &cs->css;

free_cpus:
 free_cpumask_var(cs->cpus_allowed);
free_cs:
 kfree(cs);
 return ERR_PTR(-ENOMEM);
}

static int cpuset_css_online(struct cgroup_subsys_state *css)
{
 struct cpuset *cs = css_cs(css);
 struct cpuset *parent = parent_cs(cs);
 struct cpuset *tmp_cs;
 struct cgroup_subsys_state *pos_css;

 if (!parent)
  return 0;

 mutex_lock(&cpuset_mutex);

 set_bit(CS_ONLINE, &cs->flags);
 if (is_spread_page(parent))
  set_bit(CS_SPREAD_PAGE, &cs->flags);
 if (is_spread_slab(parent))
  set_bit(CS_SPREAD_SLAB, &cs->flags);

 cpuset_inc();

 spin_lock_irq(&callback_lock);
 if (cgroup_subsys_on_dfl(cpuset_cgrp_subsys)) {
  cpumask_copy(cs->effective_cpus, parent->effective_cpus);
  cs->effective_mems = parent->effective_mems;
 }
 spin_unlock_irq(&callback_lock);

 if (!test_bit(CGRP_CPUSET_CLONE_CHILDREN, &css->cgroup->flags))
  goto out_unlock;
 rcu_read_lock();
 cpuset_for_each_child(tmp_cs, pos_css, parent) {
  if (is_mem_exclusive(tmp_cs) || is_cpu_exclusive(tmp_cs)) {
   rcu_read_unlock();
   goto out_unlock;
  }
 }
 rcu_read_unlock();

 spin_lock_irq(&callback_lock);
 cs->mems_allowed = parent->mems_allowed;
 cs->effective_mems = parent->mems_allowed;
 cpumask_copy(cs->cpus_allowed, parent->cpus_allowed);
 cpumask_copy(cs->effective_cpus, parent->cpus_allowed);
 spin_unlock_irq(&callback_lock);
out_unlock:
 mutex_unlock(&cpuset_mutex);
 return 0;
}







static void cpuset_css_offline(struct cgroup_subsys_state *css)
{
 struct cpuset *cs = css_cs(css);

 mutex_lock(&cpuset_mutex);

 if (is_sched_load_balance(cs))
  update_flag(CS_SCHED_LOAD_BALANCE, cs, 0);

 cpuset_dec();
 clear_bit(CS_ONLINE, &cs->flags);

 mutex_unlock(&cpuset_mutex);
}

static void cpuset_css_free(struct cgroup_subsys_state *css)
{
 struct cpuset *cs = css_cs(css);

 free_cpumask_var(cs->effective_cpus);
 free_cpumask_var(cs->cpus_allowed);
 kfree(cs);
}

static void cpuset_bind(struct cgroup_subsys_state *root_css)
{
 mutex_lock(&cpuset_mutex);
 spin_lock_irq(&callback_lock);

 if (cgroup_subsys_on_dfl(cpuset_cgrp_subsys)) {
  cpumask_copy(top_cpuset.cpus_allowed, cpu_possible_mask);
  top_cpuset.mems_allowed = node_possible_map;
 } else {
  cpumask_copy(top_cpuset.cpus_allowed,
        top_cpuset.effective_cpus);
  top_cpuset.mems_allowed = top_cpuset.effective_mems;
 }

 spin_unlock_irq(&callback_lock);
 mutex_unlock(&cpuset_mutex);
}

struct cgroup_subsys cpuset_cgrp_subsys = {
 .css_alloc = cpuset_css_alloc,
 .css_online = cpuset_css_online,
 .css_offline = cpuset_css_offline,
 .css_free = cpuset_css_free,
 .can_attach = cpuset_can_attach,
 .cancel_attach = cpuset_cancel_attach,
 .attach = cpuset_attach,
 .post_attach = cpuset_post_attach,
 .bind = cpuset_bind,
 .legacy_cftypes = files,
 .early_init = true,
};







int __init cpuset_init(void)
{
 int err = 0;

 if (!alloc_cpumask_var(&top_cpuset.cpus_allowed, GFP_KERNEL))
  BUG();
 if (!alloc_cpumask_var(&top_cpuset.effective_cpus, GFP_KERNEL))
  BUG();

 cpumask_setall(top_cpuset.cpus_allowed);
 nodes_setall(top_cpuset.mems_allowed);
 cpumask_setall(top_cpuset.effective_cpus);
 nodes_setall(top_cpuset.effective_mems);

 fmeter_init(&top_cpuset.fmeter);
 set_bit(CS_SCHED_LOAD_BALANCE, &top_cpuset.flags);
 top_cpuset.relax_domain_level = -1;

 err = register_filesystem(&cpuset_fs_type);
 if (err < 0)
  return err;

 if (!alloc_cpumask_var(&cpus_attach, GFP_KERNEL))
  BUG();

 return 0;
}
static void remove_tasks_in_empty_cpuset(struct cpuset *cs)
{
 struct cpuset *parent;





 parent = parent_cs(cs);
 while (cpumask_empty(parent->cpus_allowed) ||
   nodes_empty(parent->mems_allowed))
  parent = parent_cs(parent);

 if (cgroup_transfer_tasks(parent->css.cgroup, cs->css.cgroup)) {
  pr_err("cpuset: failed to transfer tasks out of empty cpuset ");
  pr_cont_cgroup_name(cs->css.cgroup);
  pr_cont("\n");
 }
}

static void
hotplug_update_tasks_legacy(struct cpuset *cs,
       struct cpumask *new_cpus, nodemask_t *new_mems,
       bool cpus_updated, bool mems_updated)
{
 bool is_empty;

 spin_lock_irq(&callback_lock);
 cpumask_copy(cs->cpus_allowed, new_cpus);
 cpumask_copy(cs->effective_cpus, new_cpus);
 cs->mems_allowed = *new_mems;
 cs->effective_mems = *new_mems;
 spin_unlock_irq(&callback_lock);





 if (cpus_updated && !cpumask_empty(cs->cpus_allowed))
  update_tasks_cpumask(cs);
 if (mems_updated && !nodes_empty(cs->mems_allowed))
  update_tasks_nodemask(cs);

 is_empty = cpumask_empty(cs->cpus_allowed) ||
     nodes_empty(cs->mems_allowed);

 mutex_unlock(&cpuset_mutex);






 if (is_empty)
  remove_tasks_in_empty_cpuset(cs);

 mutex_lock(&cpuset_mutex);
}

static void
hotplug_update_tasks(struct cpuset *cs,
       struct cpumask *new_cpus, nodemask_t *new_mems,
       bool cpus_updated, bool mems_updated)
{
 if (cpumask_empty(new_cpus))
  cpumask_copy(new_cpus, parent_cs(cs)->effective_cpus);
 if (nodes_empty(*new_mems))
  *new_mems = parent_cs(cs)->effective_mems;

 spin_lock_irq(&callback_lock);
 cpumask_copy(cs->effective_cpus, new_cpus);
 cs->effective_mems = *new_mems;
 spin_unlock_irq(&callback_lock);

 if (cpus_updated)
  update_tasks_cpumask(cs);
 if (mems_updated)
  update_tasks_nodemask(cs);
}
static void cpuset_hotplug_update_tasks(struct cpuset *cs)
{
 static cpumask_t new_cpus;
 static nodemask_t new_mems;
 bool cpus_updated;
 bool mems_updated;
retry:
 wait_event(cpuset_attach_wq, cs->attach_in_progress == 0);

 mutex_lock(&cpuset_mutex);





 if (cs->attach_in_progress) {
  mutex_unlock(&cpuset_mutex);
  goto retry;
 }

 cpumask_and(&new_cpus, cs->cpus_allowed, parent_cs(cs)->effective_cpus);
 nodes_and(new_mems, cs->mems_allowed, parent_cs(cs)->effective_mems);

 cpus_updated = !cpumask_equal(&new_cpus, cs->effective_cpus);
 mems_updated = !nodes_equal(new_mems, cs->effective_mems);

 if (cgroup_subsys_on_dfl(cpuset_cgrp_subsys))
  hotplug_update_tasks(cs, &new_cpus, &new_mems,
         cpus_updated, mems_updated);
 else
  hotplug_update_tasks_legacy(cs, &new_cpus, &new_mems,
         cpus_updated, mems_updated);

 mutex_unlock(&cpuset_mutex);
}
static void cpuset_hotplug_workfn(struct work_struct *work)
{
 static cpumask_t new_cpus;
 static nodemask_t new_mems;
 bool cpus_updated, mems_updated;
 bool on_dfl = cgroup_subsys_on_dfl(cpuset_cgrp_subsys);

 mutex_lock(&cpuset_mutex);


 cpumask_copy(&new_cpus, cpu_active_mask);
 new_mems = node_states[N_MEMORY];

 cpus_updated = !cpumask_equal(top_cpuset.effective_cpus, &new_cpus);
 mems_updated = !nodes_equal(top_cpuset.effective_mems, new_mems);


 if (cpus_updated) {
  spin_lock_irq(&callback_lock);
  if (!on_dfl)
   cpumask_copy(top_cpuset.cpus_allowed, &new_cpus);
  cpumask_copy(top_cpuset.effective_cpus, &new_cpus);
  spin_unlock_irq(&callback_lock);

 }


 if (mems_updated) {
  spin_lock_irq(&callback_lock);
  if (!on_dfl)
   top_cpuset.mems_allowed = new_mems;
  top_cpuset.effective_mems = new_mems;
  spin_unlock_irq(&callback_lock);
  update_tasks_nodemask(&top_cpuset);
 }

 mutex_unlock(&cpuset_mutex);


 if (cpus_updated || mems_updated) {
  struct cpuset *cs;
  struct cgroup_subsys_state *pos_css;

  rcu_read_lock();
  cpuset_for_each_descendant_pre(cs, pos_css, &top_cpuset) {
   if (cs == &top_cpuset || !css_tryget_online(&cs->css))
    continue;
   rcu_read_unlock();

   cpuset_hotplug_update_tasks(cs);

   rcu_read_lock();
   css_put(&cs->css);
  }
  rcu_read_unlock();
 }


 if (cpus_updated)
  rebuild_sched_domains();
}

void cpuset_update_active_cpus(bool cpu_online)
{
 partition_sched_domains(1, NULL, NULL);
 schedule_work(&cpuset_hotplug_work);
}






static int cpuset_track_online_nodes(struct notifier_block *self,
    unsigned long action, void *arg)
{
 schedule_work(&cpuset_hotplug_work);
 return NOTIFY_OK;
}

static struct notifier_block cpuset_track_online_nodes_nb = {
 .notifier_call = cpuset_track_online_nodes,
 .priority = 10,
};






void __init cpuset_init_smp(void)
{
 cpumask_copy(top_cpuset.cpus_allowed, cpu_active_mask);
 top_cpuset.mems_allowed = node_states[N_MEMORY];
 top_cpuset.old_mems_allowed = top_cpuset.mems_allowed;

 cpumask_copy(top_cpuset.effective_cpus, cpu_active_mask);
 top_cpuset.effective_mems = node_states[N_MEMORY];

 register_hotmemory_notifier(&cpuset_track_online_nodes_nb);

 cpuset_migrate_mm_wq = alloc_ordered_workqueue("cpuset_migrate_mm", 0);
 BUG_ON(!cpuset_migrate_mm_wq);
}
void cpuset_cpus_allowed(struct task_struct *tsk, struct cpumask *pmask)
{
 unsigned long flags;

 spin_lock_irqsave(&callback_lock, flags);
 rcu_read_lock();
 guarantee_online_cpus(task_cs(tsk), pmask);
 rcu_read_unlock();
 spin_unlock_irqrestore(&callback_lock, flags);
}

void cpuset_cpus_allowed_fallback(struct task_struct *tsk)
{
 rcu_read_lock();
 do_set_cpus_allowed(tsk, task_cs(tsk)->effective_cpus);
 rcu_read_unlock();
}

void __init cpuset_init_current_mems_allowed(void)
{
 nodes_setall(current->mems_allowed);
}
nodemask_t cpuset_mems_allowed(struct task_struct *tsk)
{
 nodemask_t mask;
 unsigned long flags;

 spin_lock_irqsave(&callback_lock, flags);
 rcu_read_lock();
 guarantee_online_mems(task_cs(tsk), &mask);
 rcu_read_unlock();
 spin_unlock_irqrestore(&callback_lock, flags);

 return mask;
}







int cpuset_nodemask_valid_mems_allowed(nodemask_t *nodemask)
{
 return nodes_intersects(*nodemask, current->mems_allowed);
}







static struct cpuset *nearest_hardwall_ancestor(struct cpuset *cs)
{
 while (!(is_mem_exclusive(cs) || is_mem_hardwall(cs)) && parent_cs(cs))
  cs = parent_cs(cs);
 return cs;
}
bool __cpuset_node_allowed(int node, gfp_t gfp_mask)
{
 struct cpuset *cs;
 int allowed;
 unsigned long flags;

 if (in_interrupt())
  return true;
 if (node_isset(node, current->mems_allowed))
  return true;




 if (unlikely(test_thread_flag(TIF_MEMDIE)))
  return true;
 if (gfp_mask & __GFP_HARDWALL)
  return false;

 if (current->flags & PF_EXITING)
  return true;


 spin_lock_irqsave(&callback_lock, flags);

 rcu_read_lock();
 cs = nearest_hardwall_ancestor(task_cs(current));
 allowed = node_isset(node, cs->mems_allowed);
 rcu_read_unlock();

 spin_unlock_irqrestore(&callback_lock, flags);
 return allowed;
}
static int cpuset_spread_node(int *rotor)
{
 return *rotor = next_node_in(*rotor, current->mems_allowed);
}

int cpuset_mem_spread_node(void)
{
 if (current->cpuset_mem_spread_rotor == NUMA_NO_NODE)
  current->cpuset_mem_spread_rotor =
   node_random(&current->mems_allowed);

 return cpuset_spread_node(&current->cpuset_mem_spread_rotor);
}

int cpuset_slab_spread_node(void)
{
 if (current->cpuset_slab_spread_rotor == NUMA_NO_NODE)
  current->cpuset_slab_spread_rotor =
   node_random(&current->mems_allowed);

 return cpuset_spread_node(&current->cpuset_slab_spread_rotor);
}

EXPORT_SYMBOL_GPL(cpuset_mem_spread_node);
int cpuset_mems_allowed_intersects(const struct task_struct *tsk1,
       const struct task_struct *tsk2)
{
 return nodes_intersects(tsk1->mems_allowed, tsk2->mems_allowed);
}







void cpuset_print_current_mems_allowed(void)
{
 struct cgroup *cgrp;

 rcu_read_lock();

 cgrp = task_cs(current)->css.cgroup;
 pr_info("%s cpuset=", current->comm);
 pr_cont_cgroup_name(cgrp);
 pr_cont(" mems_allowed=%*pbl\n",
  nodemask_pr_args(&current->mems_allowed));

 rcu_read_unlock();
}







int cpuset_memory_pressure_enabled __read_mostly;
void __cpuset_memory_pressure_bump(void)
{
 rcu_read_lock();
 fmeter_markevent(&task_cs(current)->fmeter);
 rcu_read_unlock();
}

int proc_cpuset_show(struct seq_file *m, struct pid_namespace *ns,
       struct pid *pid, struct task_struct *tsk)
{
 char *buf, *p;
 struct cgroup_subsys_state *css;
 int retval;

 retval = -ENOMEM;
 buf = kmalloc(PATH_MAX, GFP_KERNEL);
 if (!buf)
  goto out;

 retval = -ENAMETOOLONG;
 css = task_get_css(tsk, cpuset_cgrp_id);
 p = cgroup_path_ns(css->cgroup, buf, PATH_MAX,
      current->nsproxy->cgroup_ns);
 css_put(css);
 if (!p)
  goto out_free;
 seq_puts(m, p);
 seq_putc(m, '\n');
 retval = 0;
out_free:
 kfree(buf);
out:
 return retval;
}


void cpuset_task_status_allowed(struct seq_file *m, struct task_struct *task)
{
 seq_printf(m, "Mems_allowed:\t%*pb\n",
     nodemask_pr_args(&task->mems_allowed));
 seq_printf(m, "Mems_allowed_list:\t%*pbl\n",
     nodemask_pr_args(&task->mems_allowed));
}





unsigned long saved_max_pfn;
unsigned long long elfcorehdr_addr = ELFCORE_ADDR_MAX;
EXPORT_SYMBOL_GPL(elfcorehdr_addr);




unsigned long long elfcorehdr_size;







static int __init setup_elfcorehdr(char *arg)
{
 char *end;
 if (!arg)
  return -EINVAL;
 elfcorehdr_addr = memparse(arg, &end);
 if (*end == '@') {
  elfcorehdr_size = elfcorehdr_addr;
  elfcorehdr_addr = memparse(end + 1, &end);
 }
 return end > arg ? 0 : -EINVAL;
}
early_param("elfcorehdr", setup_elfcorehdr);

 printk("[%-5.5s%5u] " FMT "\n", \
        current->comm, current->pid, ##__VA_ARGS__)
do { \
 if (0) \
  no_printk("[%-5.5s%5u] " FMT "\n", \
     current->comm, current->pid, ##__VA_ARGS__); \
} while (0)

static struct kmem_cache *cred_jar;


struct group_info init_groups = { .usage = ATOMIC_INIT(2) };




struct cred init_cred = {
 .usage = ATOMIC_INIT(4),
 .subscribers = ATOMIC_INIT(2),
 .magic = CRED_MAGIC,
 .uid = GLOBAL_ROOT_UID,
 .gid = GLOBAL_ROOT_GID,
 .suid = GLOBAL_ROOT_UID,
 .sgid = GLOBAL_ROOT_GID,
 .euid = GLOBAL_ROOT_UID,
 .egid = GLOBAL_ROOT_GID,
 .fsuid = GLOBAL_ROOT_UID,
 .fsgid = GLOBAL_ROOT_GID,
 .securebits = SECUREBITS_DEFAULT,
 .cap_inheritable = CAP_EMPTY_SET,
 .cap_permitted = CAP_FULL_SET,
 .cap_effective = CAP_FULL_SET,
 .cap_bset = CAP_FULL_SET,
 .user = INIT_USER,
 .user_ns = &init_user_ns,
 .group_info = &init_groups,
};

static inline void set_cred_subscribers(struct cred *cred, int n)
{
 atomic_set(&cred->subscribers, n);
}

static inline int read_cred_subscribers(const struct cred *cred)
{
 return atomic_read(&cred->subscribers);
 return 0;
}

static inline void alter_cred_subscribers(const struct cred *_cred, int n)
{
 struct cred *cred = (struct cred *) _cred;

 atomic_add(n, &cred->subscribers);
}




static void put_cred_rcu(struct rcu_head *rcu)
{
 struct cred *cred = container_of(rcu, struct cred, rcu);

 kdebug("put_cred_rcu(%p)", cred);

 if (cred->magic != CRED_MAGIC_DEAD ||
     atomic_read(&cred->usage) != 0 ||
     read_cred_subscribers(cred) != 0)
  panic("CRED: put_cred_rcu() sees %p with"
        " mag %x, put %p, usage %d, subscr %d\n",
        cred, cred->magic, cred->put_addr,
        atomic_read(&cred->usage),
        read_cred_subscribers(cred));
 if (atomic_read(&cred->usage) != 0)
  panic("CRED: put_cred_rcu() sees %p with usage %d\n",
        cred, atomic_read(&cred->usage));

 security_cred_free(cred);
 key_put(cred->session_keyring);
 key_put(cred->process_keyring);
 key_put(cred->thread_keyring);
 key_put(cred->request_key_auth);
 if (cred->group_info)
  put_group_info(cred->group_info);
 free_uid(cred->user);
 put_user_ns(cred->user_ns);
 kmem_cache_free(cred_jar, cred);
}







void __put_cred(struct cred *cred)
{
 kdebug("__put_cred(%p{%d,%d})", cred,
        atomic_read(&cred->usage),
        read_cred_subscribers(cred));

 BUG_ON(atomic_read(&cred->usage) != 0);
 BUG_ON(read_cred_subscribers(cred) != 0);
 cred->magic = CRED_MAGIC_DEAD;
 cred->put_addr = __builtin_return_address(0);
 BUG_ON(cred == current->cred);
 BUG_ON(cred == current->real_cred);

 call_rcu(&cred->rcu, put_cred_rcu);
}
EXPORT_SYMBOL(__put_cred);




void exit_creds(struct task_struct *tsk)
{
 struct cred *cred;

 kdebug("exit_creds(%u,%p,%p,{%d,%d})", tsk->pid, tsk->real_cred, tsk->cred,
        atomic_read(&tsk->cred->usage),
        read_cred_subscribers(tsk->cred));

 cred = (struct cred *) tsk->real_cred;
 tsk->real_cred = NULL;
 validate_creds(cred);
 alter_cred_subscribers(cred, -1);
 put_cred(cred);

 cred = (struct cred *) tsk->cred;
 tsk->cred = NULL;
 validate_creds(cred);
 alter_cred_subscribers(cred, -1);
 put_cred(cred);
}
const struct cred *get_task_cred(struct task_struct *task)
{
 const struct cred *cred;

 rcu_read_lock();

 do {
  cred = __task_cred((task));
  BUG_ON(!cred);
 } while (!atomic_inc_not_zero(&((struct cred *)cred)->usage));

 rcu_read_unlock();
 return cred;
}





struct cred *cred_alloc_blank(void)
{
 struct cred *new;

 new = kmem_cache_zalloc(cred_jar, GFP_KERNEL);
 if (!new)
  return NULL;

 atomic_set(&new->usage, 1);
 new->magic = CRED_MAGIC;

 if (security_cred_alloc_blank(new, GFP_KERNEL) < 0)
  goto error;

 return new;

error:
 abort_creds(new);
 return NULL;
}
struct cred *prepare_creds(void)
{
 struct task_struct *task = current;
 const struct cred *old;
 struct cred *new;

 validate_process_creds();

 new = kmem_cache_alloc(cred_jar, GFP_KERNEL);
 if (!new)
  return NULL;

 kdebug("prepare_creds() alloc %p", new);

 old = task->cred;
 memcpy(new, old, sizeof(struct cred));

 atomic_set(&new->usage, 1);
 set_cred_subscribers(new, 0);
 get_group_info(new->group_info);
 get_uid(new->user);
 get_user_ns(new->user_ns);

 key_get(new->session_keyring);
 key_get(new->process_keyring);
 key_get(new->thread_keyring);
 key_get(new->request_key_auth);

 new->security = NULL;

 if (security_prepare_creds(new, old, GFP_KERNEL) < 0)
  goto error;
 validate_creds(new);
 return new;

error:
 abort_creds(new);
 return NULL;
}
EXPORT_SYMBOL(prepare_creds);





struct cred *prepare_exec_creds(void)
{
 struct cred *new;

 new = prepare_creds();
 if (!new)
  return new;


 key_put(new->thread_keyring);
 new->thread_keyring = NULL;


 key_put(new->process_keyring);
 new->process_keyring = NULL;

 return new;
}
int copy_creds(struct task_struct *p, unsigned long clone_flags)
{
 struct cred *new;
 int ret;

 if (
  !p->cred->thread_keyring &&
  clone_flags & CLONE_THREAD
     ) {
  p->real_cred = get_cred(p->cred);
  get_cred(p->cred);
  alter_cred_subscribers(p->cred, 2);
  kdebug("share_creds(%p{%d,%d})",
         p->cred, atomic_read(&p->cred->usage),
         read_cred_subscribers(p->cred));
  atomic_inc(&p->cred->user->processes);
  return 0;
 }

 new = prepare_creds();
 if (!new)
  return -ENOMEM;

 if (clone_flags & CLONE_NEWUSER) {
  ret = create_user_ns(new);
  if (ret < 0)
   goto error_put;
 }



 if (new->thread_keyring) {
  key_put(new->thread_keyring);
  new->thread_keyring = NULL;
  if (clone_flags & CLONE_THREAD)
   install_thread_keyring_to_cred(new);
 }




 if (!(clone_flags & CLONE_THREAD)) {
  key_put(new->process_keyring);
  new->process_keyring = NULL;
 }

 atomic_inc(&new->user->processes);
 p->cred = p->real_cred = get_cred(new);
 alter_cred_subscribers(new, 2);
 validate_creds(new);
 return 0;

error_put:
 put_cred(new);
 return ret;
}

static bool cred_cap_issubset(const struct cred *set, const struct cred *subset)
{
 const struct user_namespace *set_ns = set->user_ns;
 const struct user_namespace *subset_ns = subset->user_ns;




 if (set_ns == subset_ns)
  return cap_issubset(subset->cap_permitted, set->cap_permitted);






 for (;subset_ns != &init_user_ns; subset_ns = subset_ns->parent) {
  if ((set_ns == subset_ns->parent) &&
      uid_eq(subset_ns->owner, set->euid))
   return true;
 }

 return false;
}
int commit_creds(struct cred *new)
{
 struct task_struct *task = current;
 const struct cred *old = task->real_cred;

 kdebug("commit_creds(%p{%d,%d})", new,
        atomic_read(&new->usage),
        read_cred_subscribers(new));

 BUG_ON(task->cred != old);
 BUG_ON(read_cred_subscribers(old) < 2);
 validate_creds(old);
 validate_creds(new);
 BUG_ON(atomic_read(&new->usage) < 1);

 get_cred(new);


 if (!uid_eq(old->euid, new->euid) ||
     !gid_eq(old->egid, new->egid) ||
     !uid_eq(old->fsuid, new->fsuid) ||
     !gid_eq(old->fsgid, new->fsgid) ||
     !cred_cap_issubset(old, new)) {
  if (task->mm)
   set_dumpable(task->mm, suid_dumpable);
  task->pdeath_signal = 0;
  smp_wmb();
 }


 if (!uid_eq(new->fsuid, old->fsuid))
  key_fsuid_changed(task);
 if (!gid_eq(new->fsgid, old->fsgid))
  key_fsgid_changed(task);





 alter_cred_subscribers(new, 2);
 if (new->user != old->user)
  atomic_inc(&new->user->processes);
 rcu_assign_pointer(task->real_cred, new);
 rcu_assign_pointer(task->cred, new);
 if (new->user != old->user)
  atomic_dec(&old->user->processes);
 alter_cred_subscribers(old, -2);


 if (!uid_eq(new->uid, old->uid) ||
     !uid_eq(new->euid, old->euid) ||
     !uid_eq(new->suid, old->suid) ||
     !uid_eq(new->fsuid, old->fsuid))
  proc_id_connector(task, PROC_EVENT_UID);

 if (!gid_eq(new->gid, old->gid) ||
     !gid_eq(new->egid, old->egid) ||
     !gid_eq(new->sgid, old->sgid) ||
     !gid_eq(new->fsgid, old->fsgid))
  proc_id_connector(task, PROC_EVENT_GID);


 put_cred(old);
 put_cred(old);
 return 0;
}
EXPORT_SYMBOL(commit_creds);
void abort_creds(struct cred *new)
{
 kdebug("abort_creds(%p{%d,%d})", new,
        atomic_read(&new->usage),
        read_cred_subscribers(new));

 BUG_ON(read_cred_subscribers(new) != 0);
 BUG_ON(atomic_read(&new->usage) < 1);
 put_cred(new);
}
EXPORT_SYMBOL(abort_creds);
const struct cred *override_creds(const struct cred *new)
{
 const struct cred *old = current->cred;

 kdebug("override_creds(%p{%d,%d})", new,
        atomic_read(&new->usage),
        read_cred_subscribers(new));

 validate_creds(old);
 validate_creds(new);
 get_cred(new);
 alter_cred_subscribers(new, 1);
 rcu_assign_pointer(current->cred, new);
 alter_cred_subscribers(old, -1);

 kdebug("override_creds() = %p{%d,%d}", old,
        atomic_read(&old->usage),
        read_cred_subscribers(old));
 return old;
}
EXPORT_SYMBOL(override_creds);
void revert_creds(const struct cred *old)
{
 const struct cred *override = current->cred;

 kdebug("revert_creds(%p{%d,%d})", old,
        atomic_read(&old->usage),
        read_cred_subscribers(old));

 validate_creds(old);
 validate_creds(override);
 alter_cred_subscribers(old, 1);
 rcu_assign_pointer(current->cred, old);
 alter_cred_subscribers(override, -1);
 put_cred(override);
}
EXPORT_SYMBOL(revert_creds);




void __init cred_init(void)
{

 cred_jar = kmem_cache_create("cred_jar", sizeof(struct cred), 0,
   SLAB_HWCACHE_ALIGN|SLAB_PANIC|SLAB_ACCOUNT, NULL);
}
struct cred *prepare_kernel_cred(struct task_struct *daemon)
{
 const struct cred *old;
 struct cred *new;

 new = kmem_cache_alloc(cred_jar, GFP_KERNEL);
 if (!new)
  return NULL;

 kdebug("prepare_kernel_cred() alloc %p", new);

 if (daemon)
  old = get_task_cred(daemon);
 else
  old = get_cred(&init_cred);

 validate_creds(old);

 *new = *old;
 atomic_set(&new->usage, 1);
 set_cred_subscribers(new, 0);
 get_uid(new->user);
 get_user_ns(new->user_ns);
 get_group_info(new->group_info);

 new->session_keyring = NULL;
 new->process_keyring = NULL;
 new->thread_keyring = NULL;
 new->request_key_auth = NULL;
 new->jit_keyring = KEY_REQKEY_DEFL_THREAD_KEYRING;

 new->security = NULL;
 if (security_prepare_creds(new, old, GFP_KERNEL) < 0)
  goto error;

 put_cred(old);
 validate_creds(new);
 return new;

error:
 put_cred(new);
 put_cred(old);
 return NULL;
}
EXPORT_SYMBOL(prepare_kernel_cred);
int set_security_override(struct cred *new, u32 secid)
{
 return security_kernel_act_as(new, secid);
}
EXPORT_SYMBOL(set_security_override);
int set_security_override_from_ctx(struct cred *new, const char *secctx)
{
 u32 secid;
 int ret;

 ret = security_secctx_to_secid(secctx, strlen(secctx), &secid);
 if (ret < 0)
  return ret;

 return set_security_override(new, secid);
}
EXPORT_SYMBOL(set_security_override_from_ctx);
int set_create_files_as(struct cred *new, struct inode *inode)
{
 new->fsuid = inode->i_uid;
 new->fsgid = inode->i_gid;
 return security_kernel_create_files_as(new, inode);
}
EXPORT_SYMBOL(set_create_files_as);


bool creds_are_invalid(const struct cred *cred)
{
 if (cred->magic != CRED_MAGIC)
  return true;




 if (selinux_is_enabled() && cred->security) {
  if ((unsigned long) cred->security < PAGE_SIZE)
   return true;
  if ((*(u32 *)cred->security & 0xffffff00) ==
      (POISON_FREE << 24 | POISON_FREE << 16 | POISON_FREE << 8))
   return true;
 }
 return false;
}
EXPORT_SYMBOL(creds_are_invalid);




static void dump_invalid_creds(const struct cred *cred, const char *label,
          const struct task_struct *tsk)
{
 printk(KERN_ERR "CRED: %s credentials: %p %s%s%s\n",
        label, cred,
        cred == &init_cred ? "[init]" : "",
        cred == tsk->real_cred ? "[real]" : "",
        cred == tsk->cred ? "[eff]" : "");
 printk(KERN_ERR "CRED: ->magic=%x, put_addr=%p\n",
        cred->magic, cred->put_addr);
 printk(KERN_ERR "CRED: ->usage=%d, subscr=%d\n",
        atomic_read(&cred->usage),
        read_cred_subscribers(cred));
 printk(KERN_ERR "CRED: ->*uid = { %d,%d,%d,%d }\n",
  from_kuid_munged(&init_user_ns, cred->uid),
  from_kuid_munged(&init_user_ns, cred->euid),
  from_kuid_munged(&init_user_ns, cred->suid),
  from_kuid_munged(&init_user_ns, cred->fsuid));
 printk(KERN_ERR "CRED: ->*gid = { %d,%d,%d,%d }\n",
  from_kgid_munged(&init_user_ns, cred->gid),
  from_kgid_munged(&init_user_ns, cred->egid),
  from_kgid_munged(&init_user_ns, cred->sgid),
  from_kgid_munged(&init_user_ns, cred->fsgid));
 printk(KERN_ERR "CRED: ->security is %p\n", cred->security);
 if ((unsigned long) cred->security >= PAGE_SIZE &&
     (((unsigned long) cred->security & 0xffffff00) !=
      (POISON_FREE << 24 | POISON_FREE << 16 | POISON_FREE << 8)))
  printk(KERN_ERR "CRED: ->security {%x, %x}\n",
         ((u32*)cred->security)[0],
         ((u32*)cred->security)[1]);
}




void __invalid_creds(const struct cred *cred, const char *file, unsigned line)
{
 printk(KERN_ERR "CRED: Invalid credentials\n");
 printk(KERN_ERR "CRED: At %s:%u\n", file, line);
 dump_invalid_creds(cred, "Specified", current);
 BUG();
}
EXPORT_SYMBOL(__invalid_creds);




void __validate_process_creds(struct task_struct *tsk,
         const char *file, unsigned line)
{
 if (tsk->cred == tsk->real_cred) {
  if (unlikely(read_cred_subscribers(tsk->cred) < 2 ||
        creds_are_invalid(tsk->cred)))
   goto invalid_creds;
 } else {
  if (unlikely(read_cred_subscribers(tsk->real_cred) < 1 ||
        read_cred_subscribers(tsk->cred) < 1 ||
        creds_are_invalid(tsk->real_cred) ||
        creds_are_invalid(tsk->cred)))
   goto invalid_creds;
 }
 return;

invalid_creds:
 printk(KERN_ERR "CRED: Invalid process credentials\n");
 printk(KERN_ERR "CRED: At %s:%u\n", file, line);

 dump_invalid_creds(tsk->real_cred, "Real", tsk);
 if (tsk->cred != tsk->real_cred)
  dump_invalid_creds(tsk->cred, "Effective", tsk);
 else
  printk(KERN_ERR "CRED: Effective creds == Real creds\n");
 BUG();
}
EXPORT_SYMBOL(__validate_process_creds);




void validate_creds_for_do_exit(struct task_struct *tsk)
{
 kdebug("validate_creds_for_do_exit(%p,%p{%d,%d})",
        tsk->real_cred, tsk->cred,
        atomic_read(&tsk->cred->usage),
        read_cred_subscribers(tsk->cred));

 __validate_process_creds(tsk, __FILE__, __LINE__);
}





static int kgdb_break_asap;

struct debuggerinfo_struct kgdb_info[NR_CPUS];




int kgdb_connected;
EXPORT_SYMBOL_GPL(kgdb_connected);


int kgdb_io_module_registered;


static int exception_level;

struct kgdb_io *dbg_io_ops;
static DEFINE_SPINLOCK(kgdb_registration_lock);


static int kgdbreboot;

static int kgdb_con_registered;

static int kgdb_use_con;

bool dbg_is_early = true;

int dbg_switch_cpu;


int dbg_kdb_mode = 1;

static int __init opt_kgdb_con(char *str)
{
 kgdb_use_con = 1;
 return 0;
}

early_param("kgdbcon", opt_kgdb_con);

module_param(kgdb_use_con, int, 0644);
module_param(kgdbreboot, int, 0644);





static struct kgdb_bkpt kgdb_break[KGDB_MAX_BREAKPOINTS] = {
 [0 ... KGDB_MAX_BREAKPOINTS-1] = { .state = BP_UNDEFINED }
};




atomic_t kgdb_active = ATOMIC_INIT(-1);
EXPORT_SYMBOL_GPL(kgdb_active);
static DEFINE_RAW_SPINLOCK(dbg_master_lock);
static DEFINE_RAW_SPINLOCK(dbg_slave_lock);





static atomic_t masters_in_kgdb;
static atomic_t slaves_in_kgdb;
static atomic_t kgdb_break_tasklet_var;
atomic_t kgdb_setting_breakpoint;

struct task_struct *kgdb_usethread;
struct task_struct *kgdb_contthread;

int kgdb_single_step;
static pid_t kgdb_sstep_pid;


atomic_t kgdb_cpu_doing_single_step = ATOMIC_INIT(-1);
static int kgdb_do_roundup = 1;

static int __init opt_nokgdbroundup(char *str)
{
 kgdb_do_roundup = 0;

 return 0;
}

early_param("nokgdbroundup", opt_nokgdbroundup);
int __weak kgdb_arch_set_breakpoint(struct kgdb_bkpt *bpt)
{
 int err;

 err = probe_kernel_read(bpt->saved_instr, (char *)bpt->bpt_addr,
    BREAK_INSTR_SIZE);
 if (err)
  return err;
 err = probe_kernel_write((char *)bpt->bpt_addr,
     arch_kgdb_ops.gdb_bpt_instr, BREAK_INSTR_SIZE);
 return err;
}

int __weak kgdb_arch_remove_breakpoint(struct kgdb_bkpt *bpt)
{
 return probe_kernel_write((char *)bpt->bpt_addr,
      (char *)bpt->saved_instr, BREAK_INSTR_SIZE);
}

int __weak kgdb_validate_break_address(unsigned long addr)
{
 struct kgdb_bkpt tmp;
 int err;





 tmp.bpt_addr = addr;
 err = kgdb_arch_set_breakpoint(&tmp);
 if (err)
  return err;
 err = kgdb_arch_remove_breakpoint(&tmp);
 if (err)
  pr_err("Critical breakpoint error, kernel memory destroyed at: %lx\n",
         addr);
 return err;
}

unsigned long __weak kgdb_arch_pc(int exception, struct pt_regs *regs)
{
 return instruction_pointer(regs);
}

int __weak kgdb_arch_init(void)
{
 return 0;
}

int __weak kgdb_skipexception(int exception, struct pt_regs *regs)
{
 return 0;
}





static void kgdb_flush_swbreak_addr(unsigned long addr)
{
 if (!CACHE_FLUSH_IS_SAFE)
  return;

 if (current->mm) {
  int i;

  for (i = 0; i < VMACACHE_SIZE; i++) {
   if (!current->vmacache[i])
    continue;
   flush_cache_range(current->vmacache[i],
       addr, addr + BREAK_INSTR_SIZE);
  }
 }


 flush_icache_range(addr, addr + BREAK_INSTR_SIZE);
}




int dbg_activate_sw_breakpoints(void)
{
 int error;
 int ret = 0;
 int i;

 for (i = 0; i < KGDB_MAX_BREAKPOINTS; i++) {
  if (kgdb_break[i].state != BP_SET)
   continue;

  error = kgdb_arch_set_breakpoint(&kgdb_break[i]);
  if (error) {
   ret = error;
   pr_info("BP install failed: %lx\n",
    kgdb_break[i].bpt_addr);
   continue;
  }

  kgdb_flush_swbreak_addr(kgdb_break[i].bpt_addr);
  kgdb_break[i].state = BP_ACTIVE;
 }
 return ret;
}

int dbg_set_sw_break(unsigned long addr)
{
 int err = kgdb_validate_break_address(addr);
 int breakno = -1;
 int i;

 if (err)
  return err;

 for (i = 0; i < KGDB_MAX_BREAKPOINTS; i++) {
  if ((kgdb_break[i].state == BP_SET) &&
     (kgdb_break[i].bpt_addr == addr))
   return -EEXIST;
 }
 for (i = 0; i < KGDB_MAX_BREAKPOINTS; i++) {
  if (kgdb_break[i].state == BP_REMOVED &&
     kgdb_break[i].bpt_addr == addr) {
   breakno = i;
   break;
  }
 }

 if (breakno == -1) {
  for (i = 0; i < KGDB_MAX_BREAKPOINTS; i++) {
   if (kgdb_break[i].state == BP_UNDEFINED) {
    breakno = i;
    break;
   }
  }
 }

 if (breakno == -1)
  return -E2BIG;

 kgdb_break[breakno].state = BP_SET;
 kgdb_break[breakno].type = BP_BREAKPOINT;
 kgdb_break[breakno].bpt_addr = addr;

 return 0;
}

int dbg_deactivate_sw_breakpoints(void)
{
 int error;
 int ret = 0;
 int i;

 for (i = 0; i < KGDB_MAX_BREAKPOINTS; i++) {
  if (kgdb_break[i].state != BP_ACTIVE)
   continue;
  error = kgdb_arch_remove_breakpoint(&kgdb_break[i]);
  if (error) {
   pr_info("BP remove failed: %lx\n",
    kgdb_break[i].bpt_addr);
   ret = error;
  }

  kgdb_flush_swbreak_addr(kgdb_break[i].bpt_addr);
  kgdb_break[i].state = BP_SET;
 }
 return ret;
}

int dbg_remove_sw_break(unsigned long addr)
{
 int i;

 for (i = 0; i < KGDB_MAX_BREAKPOINTS; i++) {
  if ((kgdb_break[i].state == BP_SET) &&
    (kgdb_break[i].bpt_addr == addr)) {
   kgdb_break[i].state = BP_REMOVED;
   return 0;
  }
 }
 return -ENOENT;
}

int kgdb_isremovedbreak(unsigned long addr)
{
 int i;

 for (i = 0; i < KGDB_MAX_BREAKPOINTS; i++) {
  if ((kgdb_break[i].state == BP_REMOVED) &&
     (kgdb_break[i].bpt_addr == addr))
   return 1;
 }
 return 0;
}

int dbg_remove_all_break(void)
{
 int error;
 int i;


 for (i = 0; i < KGDB_MAX_BREAKPOINTS; i++) {
  if (kgdb_break[i].state != BP_ACTIVE)
   goto setundefined;
  error = kgdb_arch_remove_breakpoint(&kgdb_break[i]);
  if (error)
   pr_err("breakpoint remove failed: %lx\n",
          kgdb_break[i].bpt_addr);
setundefined:
  kgdb_break[i].state = BP_UNDEFINED;
 }


 if (arch_kgdb_ops.remove_all_hw_break)
  arch_kgdb_ops.remove_all_hw_break();

 return 0;
}
static int kgdb_io_ready(int print_wait)
{
 if (!dbg_io_ops)
  return 0;
 if (kgdb_connected)
  return 1;
 if (atomic_read(&kgdb_setting_breakpoint))
  return 1;
 if (print_wait) {
  if (!dbg_kdb_mode)
   pr_crit("waiting... or $3#33 for KDB\n");
  pr_crit("Waiting for remote debugger\n");
 }
 return 1;
}

static int kgdb_reenter_check(struct kgdb_state *ks)
{
 unsigned long addr;

 if (atomic_read(&kgdb_active) != raw_smp_processor_id())
  return 0;


 exception_level++;
 addr = kgdb_arch_pc(ks->ex_vector, ks->linux_regs);
 dbg_deactivate_sw_breakpoints();







 if (dbg_remove_sw_break(addr) == 0) {
  exception_level = 0;
  kgdb_skipexception(ks->ex_vector, ks->linux_regs);
  dbg_activate_sw_breakpoints();
  pr_crit("re-enter error: breakpoint removed %lx\n", addr);
  WARN_ON_ONCE(1);

  return 1;
 }
 dbg_remove_all_break();
 kgdb_skipexception(ks->ex_vector, ks->linux_regs);

 if (exception_level > 1) {
  dump_stack();
  panic("Recursive entry to debugger");
 }

 pr_crit("re-enter exception: ALL breakpoints killed\n");

 return 0;
 dump_stack();
 panic("Recursive entry to debugger");

 return 1;
}

static void dbg_touch_watchdogs(void)
{
 touch_softlockup_watchdog_sync();
 clocksource_touch_watchdog();
 rcu_cpu_stall_reset();
}

static int kgdb_cpu_enter(struct kgdb_state *ks, struct pt_regs *regs,
  int exception_state)
{
 unsigned long flags;
 int sstep_tries = 100;
 int error;
 int cpu;
 int trace_on = 0;
 int online_cpus = num_online_cpus();
 u64 time_left;

 kgdb_info[ks->cpu].enter_kgdb++;
 kgdb_info[ks->cpu].exception_state |= exception_state;

 if (exception_state == DCPU_WANT_MASTER)
  atomic_inc(&masters_in_kgdb);
 else
  atomic_inc(&slaves_in_kgdb);

 if (arch_kgdb_ops.disable_hw_break)
  arch_kgdb_ops.disable_hw_break(regs);

acquirelock:




 local_irq_save(flags);

 cpu = ks->cpu;
 kgdb_info[cpu].debuggerinfo = regs;
 kgdb_info[cpu].task = current;
 kgdb_info[cpu].ret_state = 0;
 kgdb_info[cpu].irq_depth = hardirq_count() >> HARDIRQ_SHIFT;


 smp_mb();

 if (exception_level == 1) {
  if (raw_spin_trylock(&dbg_master_lock))
   atomic_xchg(&kgdb_active, cpu);
  goto cpu_master_loop;
 }





 while (1) {
cpu_loop:
  if (kgdb_info[cpu].exception_state & DCPU_NEXT_MASTER) {
   kgdb_info[cpu].exception_state &= ~DCPU_NEXT_MASTER;
   goto cpu_master_loop;
  } else if (kgdb_info[cpu].exception_state & DCPU_WANT_MASTER) {
   if (raw_spin_trylock(&dbg_master_lock)) {
    atomic_xchg(&kgdb_active, cpu);
    break;
   }
  } else if (kgdb_info[cpu].exception_state & DCPU_IS_SLAVE) {
   if (!raw_spin_is_locked(&dbg_slave_lock))
    goto return_normal;
  } else {
return_normal:



   if (arch_kgdb_ops.correct_hw_break)
    arch_kgdb_ops.correct_hw_break();
   if (trace_on)
    tracing_on();
   kgdb_info[cpu].exception_state &=
    ~(DCPU_WANT_MASTER | DCPU_IS_SLAVE);
   kgdb_info[cpu].enter_kgdb--;
   smp_mb__before_atomic();
   atomic_dec(&slaves_in_kgdb);
   dbg_touch_watchdogs();
   local_irq_restore(flags);
   return 0;
  }
  cpu_relax();
 }







 if (atomic_read(&kgdb_cpu_doing_single_step) != -1 &&
     (kgdb_info[cpu].task &&
      kgdb_info[cpu].task->pid != kgdb_sstep_pid) && --sstep_tries) {
  atomic_set(&kgdb_active, -1);
  raw_spin_unlock(&dbg_master_lock);
  dbg_touch_watchdogs();
  local_irq_restore(flags);

  goto acquirelock;
 }

 if (!kgdb_io_ready(1)) {
  kgdb_info[cpu].ret_state = 1;
  goto kgdb_restore;
 }




 if (kgdb_skipexception(ks->ex_vector, ks->linux_regs))
  goto kgdb_restore;


 if (dbg_io_ops->pre_exception)
  dbg_io_ops->pre_exception();





 if (!kgdb_single_step)
  raw_spin_lock(&dbg_slave_lock);


 if (ks->send_ready)
  atomic_set(ks->send_ready, 1);


 else if ((!kgdb_single_step) && kgdb_do_roundup)
  kgdb_roundup_cpus(flags);




 time_left = loops_per_jiffy * HZ;
 while (kgdb_do_roundup && --time_left &&
        (atomic_read(&masters_in_kgdb) + atomic_read(&slaves_in_kgdb)) !=
     online_cpus)
  cpu_relax();
 if (!time_left)
  pr_crit("Timed out waiting for secondary CPUs.\n");





 dbg_deactivate_sw_breakpoints();
 kgdb_single_step = 0;
 kgdb_contthread = current;
 exception_level = 0;
 trace_on = tracing_is_on();
 if (trace_on)
  tracing_off();

 while (1) {
cpu_master_loop:
  if (dbg_kdb_mode) {
   kgdb_connected = 1;
   error = kdb_stub(ks);
   if (error == -1)
    continue;
   kgdb_connected = 0;
  } else {
   error = gdb_serial_stub(ks);
  }

  if (error == DBG_PASS_EVENT) {
   dbg_kdb_mode = !dbg_kdb_mode;
  } else if (error == DBG_SWITCH_CPU_EVENT) {
   kgdb_info[dbg_switch_cpu].exception_state |=
    DCPU_NEXT_MASTER;
   goto cpu_loop;
  } else {
   kgdb_info[cpu].ret_state = error;
   break;
  }
 }


 if (dbg_io_ops->post_exception)
  dbg_io_ops->post_exception();

 if (!kgdb_single_step) {
  raw_spin_unlock(&dbg_slave_lock);

  while (kgdb_do_roundup && atomic_read(&slaves_in_kgdb))
   cpu_relax();
 }

kgdb_restore:
 if (atomic_read(&kgdb_cpu_doing_single_step) != -1) {
  int sstep_cpu = atomic_read(&kgdb_cpu_doing_single_step);
  if (kgdb_info[sstep_cpu].task)
   kgdb_sstep_pid = kgdb_info[sstep_cpu].task->pid;
  else
   kgdb_sstep_pid = 0;
 }
 if (arch_kgdb_ops.correct_hw_break)
  arch_kgdb_ops.correct_hw_break();
 if (trace_on)
  tracing_on();

 kgdb_info[cpu].exception_state &=
  ~(DCPU_WANT_MASTER | DCPU_IS_SLAVE);
 kgdb_info[cpu].enter_kgdb--;
 smp_mb__before_atomic();
 atomic_dec(&masters_in_kgdb);

 atomic_set(&kgdb_active, -1);
 raw_spin_unlock(&dbg_master_lock);
 dbg_touch_watchdogs();
 local_irq_restore(flags);

 return kgdb_info[cpu].ret_state;
}
int
kgdb_handle_exception(int evector, int signo, int ecode, struct pt_regs *regs)
{
 struct kgdb_state kgdb_var;
 struct kgdb_state *ks = &kgdb_var;
 int ret = 0;

 if (arch_kgdb_ops.enable_nmi)
  arch_kgdb_ops.enable_nmi(0);






 if (signo != SIGTRAP && panic_timeout)
  return 1;

 memset(ks, 0, sizeof(struct kgdb_state));
 ks->cpu = raw_smp_processor_id();
 ks->ex_vector = evector;
 ks->signo = signo;
 ks->err_code = ecode;
 ks->linux_regs = regs;

 if (kgdb_reenter_check(ks))
  goto out;
 if (kgdb_info[ks->cpu].enter_kgdb != 0)
  goto out;

 ret = kgdb_cpu_enter(ks, regs, DCPU_WANT_MASTER);
out:
 if (arch_kgdb_ops.enable_nmi)
  arch_kgdb_ops.enable_nmi(1);
 return ret;
}







static int module_event(struct notifier_block *self, unsigned long val,
 void *data)
{
 return 0;
}

static struct notifier_block dbg_module_load_nb = {
 .notifier_call = module_event,
};

int kgdb_nmicallback(int cpu, void *regs)
{
 struct kgdb_state kgdb_var;
 struct kgdb_state *ks = &kgdb_var;

 memset(ks, 0, sizeof(struct kgdb_state));
 ks->cpu = cpu;
 ks->linux_regs = regs;

 if (kgdb_info[ks->cpu].enter_kgdb == 0 &&
   raw_spin_is_locked(&dbg_master_lock)) {
  kgdb_cpu_enter(ks, regs, DCPU_IS_SLAVE);
  return 0;
 }
 return 1;
}

int kgdb_nmicallin(int cpu, int trapnr, void *regs, int err_code,
       atomic_t *send_ready)
{
 if (!kgdb_io_ready(0) || !send_ready)
  return 1;

 if (kgdb_info[cpu].enter_kgdb == 0) {
  struct kgdb_state kgdb_var;
  struct kgdb_state *ks = &kgdb_var;

  memset(ks, 0, sizeof(struct kgdb_state));
  ks->cpu = cpu;
  ks->ex_vector = trapnr;
  ks->signo = SIGTRAP;
  ks->err_code = err_code;
  ks->linux_regs = regs;
  ks->send_ready = send_ready;
  kgdb_cpu_enter(ks, regs, DCPU_WANT_MASTER);
  return 0;
 }
 return 1;
}

static void kgdb_console_write(struct console *co, const char *s,
   unsigned count)
{
 unsigned long flags;



 if (!kgdb_connected || atomic_read(&kgdb_active) != -1 || dbg_kdb_mode)
  return;

 local_irq_save(flags);
 gdbstub_msg_write(s, count);
 local_irq_restore(flags);
}

static struct console kgdbcons = {
 .name = "kgdb",
 .write = kgdb_console_write,
 .flags = CON_PRINTBUFFER | CON_ENABLED,
 .index = -1,
};

static void sysrq_handle_dbg(int key)
{
 if (!dbg_io_ops) {
  pr_crit("ERROR: No KGDB I/O module available\n");
  return;
 }
 if (!kgdb_connected) {
  if (!dbg_kdb_mode)
   pr_crit("KGDB or $3#33 for KDB\n");
  pr_crit("Entering KGDB\n");
 }

 kgdb_breakpoint();
}

static struct sysrq_key_op sysrq_dbg_op = {
 .handler = sysrq_handle_dbg,
 .help_msg = "debug(g)",
 .action_msg = "DEBUG",
};

static int kgdb_panic_event(struct notifier_block *self,
       unsigned long val,
       void *data)
{






 if (panic_timeout)
  return NOTIFY_DONE;

 if (dbg_kdb_mode)
  kdb_printf("PANIC: %s\n", (char *)data);
 kgdb_breakpoint();
 return NOTIFY_DONE;
}

static struct notifier_block kgdb_panic_event_nb = {
       .notifier_call = kgdb_panic_event,
       .priority = INT_MAX,
};

void __weak kgdb_arch_late(void)
{
}

void __init dbg_late_init(void)
{
 dbg_is_early = false;
 if (kgdb_io_module_registered)
  kgdb_arch_late();
 kdb_init(KDB_INIT_FULL);
}

static int
dbg_notify_reboot(struct notifier_block *this, unsigned long code, void *x)
{






 switch (kgdbreboot) {
 case 1:
  kgdb_breakpoint();
 case -1:
  goto done;
 }
 if (!dbg_kdb_mode)
  gdbstub_exit(code);
done:
 return NOTIFY_DONE;
}

static struct notifier_block dbg_reboot_notifier = {
 .notifier_call = dbg_notify_reboot,
 .next = NULL,
 .priority = INT_MAX,
};

static void kgdb_register_callbacks(void)
{
 if (!kgdb_io_module_registered) {
  kgdb_io_module_registered = 1;
  kgdb_arch_init();
  if (!dbg_is_early)
   kgdb_arch_late();
  register_module_notifier(&dbg_module_load_nb);
  register_reboot_notifier(&dbg_reboot_notifier);
  atomic_notifier_chain_register(&panic_notifier_list,
            &kgdb_panic_event_nb);
  register_sysrq_key('g', &sysrq_dbg_op);
  if (kgdb_use_con && !kgdb_con_registered) {
   register_console(&kgdbcons);
   kgdb_con_registered = 1;
  }
 }
}

static void kgdb_unregister_callbacks(void)
{





 if (kgdb_io_module_registered) {
  kgdb_io_module_registered = 0;
  unregister_reboot_notifier(&dbg_reboot_notifier);
  unregister_module_notifier(&dbg_module_load_nb);
  atomic_notifier_chain_unregister(&panic_notifier_list,
            &kgdb_panic_event_nb);
  kgdb_arch_exit();
  unregister_sysrq_key('g', &sysrq_dbg_op);
  if (kgdb_con_registered) {
   unregister_console(&kgdbcons);
   kgdb_con_registered = 0;
  }
 }
}







static void kgdb_tasklet_bpt(unsigned long ing)
{
 kgdb_breakpoint();
 atomic_set(&kgdb_break_tasklet_var, 0);
}

static DECLARE_TASKLET(kgdb_tasklet_breakpoint, kgdb_tasklet_bpt, 0);

void kgdb_schedule_breakpoint(void)
{
 if (atomic_read(&kgdb_break_tasklet_var) ||
  atomic_read(&kgdb_active) != -1 ||
  atomic_read(&kgdb_setting_breakpoint))
  return;
 atomic_inc(&kgdb_break_tasklet_var);
 tasklet_schedule(&kgdb_tasklet_breakpoint);
}
EXPORT_SYMBOL_GPL(kgdb_schedule_breakpoint);

static void kgdb_initial_breakpoint(void)
{
 kgdb_break_asap = 0;

 pr_crit("Waiting for connection from remote gdb...\n");
 kgdb_breakpoint();
}







int kgdb_register_io_module(struct kgdb_io *new_dbg_io_ops)
{
 int err;

 spin_lock(&kgdb_registration_lock);

 if (dbg_io_ops) {
  spin_unlock(&kgdb_registration_lock);

  pr_err("Another I/O driver is already registered with KGDB\n");
  return -EBUSY;
 }

 if (new_dbg_io_ops->init) {
  err = new_dbg_io_ops->init();
  if (err) {
   spin_unlock(&kgdb_registration_lock);
   return err;
  }
 }

 dbg_io_ops = new_dbg_io_ops;

 spin_unlock(&kgdb_registration_lock);

 pr_info("Registered I/O driver %s\n", new_dbg_io_ops->name);


 kgdb_register_callbacks();

 if (kgdb_break_asap)
  kgdb_initial_breakpoint();

 return 0;
}
EXPORT_SYMBOL_GPL(kgdb_register_io_module);







void kgdb_unregister_io_module(struct kgdb_io *old_dbg_io_ops)
{
 BUG_ON(kgdb_connected);





 kgdb_unregister_callbacks();

 spin_lock(&kgdb_registration_lock);

 WARN_ON_ONCE(dbg_io_ops != old_dbg_io_ops);
 dbg_io_ops = NULL;

 spin_unlock(&kgdb_registration_lock);

 pr_info("Unregistered I/O driver %s, debugger disabled\n",
  old_dbg_io_ops->name);
}
EXPORT_SYMBOL_GPL(kgdb_unregister_io_module);

int dbg_io_get_char(void)
{
 int ret = dbg_io_ops->read_char();
 if (ret == NO_POLL_CHAR)
  return -1;
 if (!dbg_kdb_mode)
  return ret;
 if (ret == 127)
  return 8;
 return ret;
}
noinline void kgdb_breakpoint(void)
{
 atomic_inc(&kgdb_setting_breakpoint);
 wmb();
 arch_kgdb_breakpoint();
 wmb();
 atomic_dec(&kgdb_setting_breakpoint);
}
EXPORT_SYMBOL_GPL(kgdb_breakpoint);

static int __init opt_kgdb_wait(char *str)
{
 kgdb_break_asap = 1;

 kdb_init(KDB_INIT_EARLY);
 if (kgdb_io_module_registered)
  kgdb_initial_breakpoint();

 return 0;
}

early_param("kgdbwait", opt_kgdb_wait);

int delayacct_on __read_mostly = 1;
EXPORT_SYMBOL_GPL(delayacct_on);
struct kmem_cache *delayacct_cache;

static int __init delayacct_setup_disable(char *str)
{
 delayacct_on = 0;
 return 1;
}
__setup("nodelayacct", delayacct_setup_disable);

void delayacct_init(void)
{
 delayacct_cache = KMEM_CACHE(task_delay_info, SLAB_PANIC|SLAB_ACCOUNT);
 delayacct_tsk_init(&init_task);
}

void __delayacct_tsk_init(struct task_struct *tsk)
{
 tsk->delays = kmem_cache_zalloc(delayacct_cache, GFP_KERNEL);
 if (tsk->delays)
  spin_lock_init(&tsk->delays->lock);
}





static void delayacct_end(u64 *start, u64 *total, u32 *count)
{
 s64 ns = ktime_get_ns() - *start;
 unsigned long flags;

 if (ns > 0) {
  spin_lock_irqsave(&current->delays->lock, flags);
  *total += ns;
  (*count)++;
  spin_unlock_irqrestore(&current->delays->lock, flags);
 }
}

void __delayacct_blkio_start(void)
{
 current->delays->blkio_start = ktime_get_ns();
}

void __delayacct_blkio_end(void)
{
 if (current->delays->flags & DELAYACCT_PF_SWAPIN)

  delayacct_end(&current->delays->blkio_start,
   &current->delays->swapin_delay,
   &current->delays->swapin_count);
 else
  delayacct_end(&current->delays->blkio_start,
   &current->delays->blkio_delay,
   &current->delays->blkio_count);
}

int __delayacct_add_tsk(struct taskstats *d, struct task_struct *tsk)
{
 cputime_t utime, stime, stimescaled, utimescaled;
 unsigned long long t2, t3;
 unsigned long flags, t1;
 s64 tmp;

 task_cputime(tsk, &utime, &stime);
 tmp = (s64)d->cpu_run_real_total;
 tmp += cputime_to_nsecs(utime + stime);
 d->cpu_run_real_total = (tmp < (s64)d->cpu_run_real_total) ? 0 : tmp;

 task_cputime_scaled(tsk, &utimescaled, &stimescaled);
 tmp = (s64)d->cpu_scaled_run_real_total;
 tmp += cputime_to_nsecs(utimescaled + stimescaled);
 d->cpu_scaled_run_real_total =
  (tmp < (s64)d->cpu_scaled_run_real_total) ? 0 : tmp;





 t1 = tsk->sched_info.pcount;
 t2 = tsk->sched_info.run_delay;
 t3 = tsk->se.sum_exec_runtime;

 d->cpu_count += t1;

 tmp = (s64)d->cpu_delay_total + t2;
 d->cpu_delay_total = (tmp < (s64)d->cpu_delay_total) ? 0 : tmp;

 tmp = (s64)d->cpu_run_virtual_total + t3;
 d->cpu_run_virtual_total =
  (tmp < (s64)d->cpu_run_virtual_total) ? 0 : tmp;



 spin_lock_irqsave(&tsk->delays->lock, flags);
 tmp = d->blkio_delay_total + tsk->delays->blkio_delay;
 d->blkio_delay_total = (tmp < d->blkio_delay_total) ? 0 : tmp;
 tmp = d->swapin_delay_total + tsk->delays->swapin_delay;
 d->swapin_delay_total = (tmp < d->swapin_delay_total) ? 0 : tmp;
 tmp = d->freepages_delay_total + tsk->delays->freepages_delay;
 d->freepages_delay_total = (tmp < d->freepages_delay_total) ? 0 : tmp;
 d->blkio_count += tsk->delays->blkio_count;
 d->swapin_count += tsk->delays->swapin_count;
 d->freepages_count += tsk->delays->freepages_count;
 spin_unlock_irqrestore(&tsk->delays->lock, flags);

 return 0;
}

__u64 __delayacct_blkio_ticks(struct task_struct *tsk)
{
 __u64 ret;
 unsigned long flags;

 spin_lock_irqsave(&tsk->delays->lock, flags);
 ret = nsec_to_clock_t(tsk->delays->blkio_delay +
    tsk->delays->swapin_delay);
 spin_unlock_irqrestore(&tsk->delays->lock, flags);
 return ret;
}

void __delayacct_freepages_start(void)
{
 current->delays->freepages_start = ktime_get_ns();
}

void __delayacct_freepages_end(void)
{
 delayacct_end(&current->delays->freepages_start,
   &current->delays->freepages_delay,
   &current->delays->freepages_count);
}





struct irq_devres {
 unsigned int irq;
 void *dev_id;
};

static void devm_irq_release(struct device *dev, void *res)
{
 struct irq_devres *this = res;

 free_irq(this->irq, this->dev_id);
}

static int devm_irq_match(struct device *dev, void *res, void *data)
{
 struct irq_devres *this = res, *match = data;

 return this->irq == match->irq && this->dev_id == match->dev_id;
}
int devm_request_threaded_irq(struct device *dev, unsigned int irq,
         irq_handler_t handler, irq_handler_t thread_fn,
         unsigned long irqflags, const char *devname,
         void *dev_id)
{
 struct irq_devres *dr;
 int rc;

 dr = devres_alloc(devm_irq_release, sizeof(struct irq_devres),
     GFP_KERNEL);
 if (!dr)
  return -ENOMEM;

 rc = request_threaded_irq(irq, handler, thread_fn, irqflags, devname,
      dev_id);
 if (rc) {
  devres_free(dr);
  return rc;
 }

 dr->irq = irq;
 dr->dev_id = dev_id;
 devres_add(dev, dr);

 return 0;
}
EXPORT_SYMBOL(devm_request_threaded_irq);
int devm_request_any_context_irq(struct device *dev, unsigned int irq,
         irq_handler_t handler, unsigned long irqflags,
         const char *devname, void *dev_id)
{
 struct irq_devres *dr;
 int rc;

 dr = devres_alloc(devm_irq_release, sizeof(struct irq_devres),
     GFP_KERNEL);
 if (!dr)
  return -ENOMEM;

 rc = request_any_context_irq(irq, handler, irqflags, devname, dev_id);
 if (rc < 0) {
  devres_free(dr);
  return rc;
 }

 dr->irq = irq;
 dr->dev_id = dev_id;
 devres_add(dev, dr);

 return rc;
}
EXPORT_SYMBOL(devm_request_any_context_irq);
void devm_free_irq(struct device *dev, unsigned int irq, void *dev_id)
{
 struct irq_devres match_data = { irq, dev_id };

 WARN_ON(devres_destroy(dev, devm_irq_release, devm_irq_match,
          &match_data));
 free_irq(irq, dev_id);
}
EXPORT_SYMBOL(devm_free_irq);
DEFINE_SPINLOCK(dma_spin_lock);












struct dma_chan {
 int lock;
 const char *device_id;
};

static struct dma_chan dma_chan_busy[MAX_DMA_CHANNELS] = {
 [4] = { 1, "cascade" },
};







int request_dma(unsigned int dmanr, const char * device_id)
{
 if (dmanr >= MAX_DMA_CHANNELS)
  return -EINVAL;

 if (xchg(&dma_chan_busy[dmanr].lock, 1) != 0)
  return -EBUSY;

 dma_chan_busy[dmanr].device_id = device_id;


 return 0;
}





void free_dma(unsigned int dmanr)
{
 if (dmanr >= MAX_DMA_CHANNELS) {
  printk(KERN_WARNING "Trying to free DMA%d\n", dmanr);
  return;
 }

 if (xchg(&dma_chan_busy[dmanr].lock, 0) == 0) {
  printk(KERN_WARNING "Trying to free free DMA%d\n", dmanr);
  return;
 }

}


int request_dma(unsigned int dmanr, const char *device_id)
{
 return -EINVAL;
}

void free_dma(unsigned int dmanr)
{
}



static int proc_dma_show(struct seq_file *m, void *v)
{
 int i;

 for (i = 0 ; i < MAX_DMA_CHANNELS ; i++) {
  if (dma_chan_busy[i].lock) {
   seq_printf(m, "%2d: %s\n", i,
       dma_chan_busy[i].device_id);
  }
 }
 return 0;
}
static int proc_dma_show(struct seq_file *m, void *v)
{
 seq_puts(m, "No DMA\n");
 return 0;
}

static int proc_dma_open(struct inode *inode, struct file *file)
{
 return single_open(file, proc_dma_show, NULL);
}

static const struct file_operations proc_dma_operations = {
 .open = proc_dma_open,
 .read = seq_read,
 .llseek = seq_lseek,
 .release = single_release,
};

static int __init proc_dma_init(void)
{
 proc_create("dma", 0, NULL, &proc_dma_operations);
 return 0;
}

__initcall(proc_dma_init);

EXPORT_SYMBOL(request_dma);
EXPORT_SYMBOL(free_dma);
EXPORT_SYMBOL(dma_spin_lock);












static void ack_bad(struct irq_data *data)
{
 struct irq_desc *desc = irq_data_to_desc(data);

 print_irq_desc(data->irq, desc);
 ack_bad_irq(data->irq);
}




static void noop(struct irq_data *data) { }

static unsigned int noop_ret(struct irq_data *data)
{
 return 0;
}




struct irq_chip no_irq_chip = {
 .name = "none",
 .irq_startup = noop_ret,
 .irq_shutdown = noop,
 .irq_enable = noop,
 .irq_disable = noop,
 .irq_ack = ack_bad,
 .flags = IRQCHIP_SKIP_SET_WAKE,
};





struct irq_chip dummy_irq_chip = {
 .name = "dummy",
 .irq_startup = noop_ret,
 .irq_shutdown = noop,
 .irq_enable = noop,
 .irq_disable = noop,
 .irq_ack = noop,
 .irq_mask = noop,
 .irq_unmask = noop,
 .flags = IRQCHIP_SKIP_SET_WAKE,
};
EXPORT_SYMBOL_GPL(dummy_irq_chip);

Elf_Half __weak elf_core_extra_phdrs(void)
{
 return 0;
}

int __weak elf_core_write_extra_phdrs(struct coredump_params *cprm, loff_t offset)
{
 return 1;
}

int __weak elf_core_write_extra_data(struct coredump_params *cprm)
{
 return 1;
}

size_t __weak elf_core_extra_data_size(void)
{
 return 0;
}

static int execdomains_proc_show(struct seq_file *m, void *v)
{
 seq_puts(m, "0-0\tLinux           \t[kernel]\n");
 return 0;
}

static int execdomains_proc_open(struct inode *inode, struct file *file)
{
 return single_open(file, execdomains_proc_show, NULL);
}

static const struct file_operations execdomains_proc_fops = {
 .open = execdomains_proc_open,
 .read = seq_read,
 .llseek = seq_lseek,
 .release = single_release,
};

static int __init proc_execdomains_init(void)
{
 proc_create("execdomains", 0, NULL, &execdomains_proc_fops);
 return 0;
}
module_init(proc_execdomains_init);

SYSCALL_DEFINE1(personality, unsigned int, personality)
{
 unsigned int old = current->personality;

 if (personality != 0xffffffff)
  set_personality(personality);

 return old;
}








static void __unhash_process(struct task_struct *p, bool group_dead)
{
 nr_threads--;
 detach_pid(p, PIDTYPE_PID);
 if (group_dead) {
  detach_pid(p, PIDTYPE_PGID);
  detach_pid(p, PIDTYPE_SID);

  list_del_rcu(&p->tasks);
  list_del_init(&p->sibling);
  __this_cpu_dec(process_counts);
 }
 list_del_rcu(&p->thread_group);
 list_del_rcu(&p->thread_node);
}




static void __exit_signal(struct task_struct *tsk)
{
 struct signal_struct *sig = tsk->signal;
 bool group_dead = thread_group_leader(tsk);
 struct sighand_struct *sighand;
 struct tty_struct *uninitialized_var(tty);
 cputime_t utime, stime;

 sighand = rcu_dereference_check(tsk->sighand,
     lockdep_tasklist_lock_is_held());
 spin_lock(&sighand->siglock);

 posix_cpu_timers_exit(tsk);
 if (group_dead) {
  posix_cpu_timers_exit_group(tsk);
  tty = sig->tty;
  sig->tty = NULL;
 } else {





  if (unlikely(has_group_leader_pid(tsk)))
   posix_cpu_timers_exit_group(tsk);





  if (sig->notify_count > 0 && !--sig->notify_count)
   wake_up_process(sig->group_exit_task);

  if (tsk == sig->curr_target)
   sig->curr_target = next_thread(tsk);
 }







 task_cputime(tsk, &utime, &stime);
 write_seqlock(&sig->stats_lock);
 sig->utime += utime;
 sig->stime += stime;
 sig->gtime += task_gtime(tsk);
 sig->min_flt += tsk->min_flt;
 sig->maj_flt += tsk->maj_flt;
 sig->nvcsw += tsk->nvcsw;
 sig->nivcsw += tsk->nivcsw;
 sig->inblock += task_io_get_inblock(tsk);
 sig->oublock += task_io_get_oublock(tsk);
 task_io_accounting_add(&sig->ioac, &tsk->ioac);
 sig->sum_sched_runtime += tsk->se.sum_exec_runtime;
 sig->nr_threads--;
 __unhash_process(tsk, group_dead);
 write_sequnlock(&sig->stats_lock);





 flush_sigqueue(&tsk->pending);
 tsk->sighand = NULL;
 spin_unlock(&sighand->siglock);

 __cleanup_sighand(sighand);
 clear_tsk_thread_flag(tsk, TIF_SIGPENDING);
 if (group_dead) {
  flush_sigqueue(&sig->shared_pending);
  tty_kref_put(tty);
 }
}

static void delayed_put_task_struct(struct rcu_head *rhp)
{
 struct task_struct *tsk = container_of(rhp, struct task_struct, rcu);

 perf_event_delayed_put(tsk);
 trace_sched_process_free(tsk);
 put_task_struct(tsk);
}


void release_task(struct task_struct *p)
{
 struct task_struct *leader;
 int zap_leader;
repeat:


 rcu_read_lock();
 atomic_dec(&__task_cred(p)->user->processes);
 rcu_read_unlock();

 proc_flush_task(p);

 write_lock_irq(&tasklist_lock);
 ptrace_release_task(p);
 __exit_signal(p);






 zap_leader = 0;
 leader = p->group_leader;
 if (leader != p && thread_group_empty(leader)
   && leader->exit_state == EXIT_ZOMBIE) {





  zap_leader = do_notify_parent(leader, leader->exit_signal);
  if (zap_leader)
   leader->exit_state = EXIT_DEAD;
 }

 write_unlock_irq(&tasklist_lock);
 release_thread(p);
 call_rcu(&p->rcu, delayed_put_task_struct);

 p = leader;
 if (unlikely(zap_leader))
  goto repeat;
}
static int will_become_orphaned_pgrp(struct pid *pgrp,
     struct task_struct *ignored_task)
{
 struct task_struct *p;

 do_each_pid_task(pgrp, PIDTYPE_PGID, p) {
  if ((p == ignored_task) ||
      (p->exit_state && thread_group_empty(p)) ||
      is_global_init(p->real_parent))
   continue;

  if (task_pgrp(p->real_parent) != pgrp &&
      task_session(p->real_parent) == task_session(p))
   return 0;
 } while_each_pid_task(pgrp, PIDTYPE_PGID, p);

 return 1;
}

int is_current_pgrp_orphaned(void)
{
 int retval;

 read_lock(&tasklist_lock);
 retval = will_become_orphaned_pgrp(task_pgrp(current), NULL);
 read_unlock(&tasklist_lock);

 return retval;
}

static bool has_stopped_jobs(struct pid *pgrp)
{
 struct task_struct *p;

 do_each_pid_task(pgrp, PIDTYPE_PGID, p) {
  if (p->signal->flags & SIGNAL_STOP_STOPPED)
   return true;
 } while_each_pid_task(pgrp, PIDTYPE_PGID, p);

 return false;
}






static void
kill_orphaned_pgrp(struct task_struct *tsk, struct task_struct *parent)
{
 struct pid *pgrp = task_pgrp(tsk);
 struct task_struct *ignored_task = tsk;

 if (!parent)



  parent = tsk->real_parent;
 else



  ignored_task = NULL;

 if (task_pgrp(parent) != pgrp &&
     task_session(parent) == task_session(tsk) &&
     will_become_orphaned_pgrp(pgrp, ignored_task) &&
     has_stopped_jobs(pgrp)) {
  __kill_pgrp_info(SIGHUP, SEND_SIG_PRIV, pgrp);
  __kill_pgrp_info(SIGCONT, SEND_SIG_PRIV, pgrp);
 }
}




void mm_update_next_owner(struct mm_struct *mm)
{
 struct task_struct *c, *g, *p = current;

retry:




 if (mm->owner != p)
  return;





 if (atomic_read(&mm->mm_users) <= 1) {
  mm->owner = NULL;
  return;
 }

 read_lock(&tasklist_lock);



 list_for_each_entry(c, &p->children, sibling) {
  if (c->mm == mm)
   goto assign_new_owner;
 }




 list_for_each_entry(c, &p->real_parent->children, sibling) {
  if (c->mm == mm)
   goto assign_new_owner;
 }




 for_each_process(g) {
  if (g->flags & PF_KTHREAD)
   continue;
  for_each_thread(g, c) {
   if (c->mm == mm)
    goto assign_new_owner;
   if (c->mm)
    break;
  }
 }
 read_unlock(&tasklist_lock);





 mm->owner = NULL;
 return;

assign_new_owner:
 BUG_ON(c == p);
 get_task_struct(c);




 task_lock(c);




 read_unlock(&tasklist_lock);
 if (c->mm != mm) {
  task_unlock(c);
  put_task_struct(c);
  goto retry;
 }
 mm->owner = c;
 task_unlock(c);
 put_task_struct(c);
}





static void exit_mm(struct task_struct *tsk)
{
 struct mm_struct *mm = tsk->mm;
 struct core_state *core_state;

 mm_release(tsk, mm);
 if (!mm)
  return;
 sync_mm_rss(mm);







 down_read(&mm->mmap_sem);
 core_state = mm->core_state;
 if (core_state) {
  struct core_thread self;

  up_read(&mm->mmap_sem);

  self.task = tsk;
  self.next = xchg(&core_state->dumper.next, &self);




  if (atomic_dec_and_test(&core_state->nr_threads))
   complete(&core_state->startup);

  for (;;) {
   set_task_state(tsk, TASK_UNINTERRUPTIBLE);
   if (!self.task)
    break;
   freezable_schedule();
  }
  __set_task_state(tsk, TASK_RUNNING);
  down_read(&mm->mmap_sem);
 }
 atomic_inc(&mm->mm_count);
 BUG_ON(mm != tsk->active_mm);

 task_lock(tsk);
 tsk->mm = NULL;
 up_read(&mm->mmap_sem);
 enter_lazy_tlb(mm, current);
 task_unlock(tsk);
 mm_update_next_owner(mm);
 mmput(mm);
 if (test_thread_flag(TIF_MEMDIE))
  exit_oom_victim(tsk);
}

static struct task_struct *find_alive_thread(struct task_struct *p)
{
 struct task_struct *t;

 for_each_thread(p, t) {
  if (!(t->flags & PF_EXITING))
   return t;
 }
 return NULL;
}

static struct task_struct *find_child_reaper(struct task_struct *father)
 __releases(&tasklist_lock)
 __acquires(&tasklist_lock)
{
 struct pid_namespace *pid_ns = task_active_pid_ns(father);
 struct task_struct *reaper = pid_ns->child_reaper;

 if (likely(reaper != father))
  return reaper;

 reaper = find_alive_thread(father);
 if (reaper) {
  pid_ns->child_reaper = reaper;
  return reaper;
 }

 write_unlock_irq(&tasklist_lock);
 if (unlikely(pid_ns == &init_pid_ns)) {
  panic("Attempted to kill init! exitcode=0x%08x\n",
   father->signal->group_exit_code ?: father->exit_code);
 }
 zap_pid_ns_processes(pid_ns);
 write_lock_irq(&tasklist_lock);

 return father;
}
static struct task_struct *find_new_reaper(struct task_struct *father,
        struct task_struct *child_reaper)
{
 struct task_struct *thread, *reaper;

 thread = find_alive_thread(father);
 if (thread)
  return thread;

 if (father->signal->has_child_subreaper) {





  for (reaper = father;
       !same_thread_group(reaper, child_reaper);
       reaper = reaper->real_parent) {

   if (reaper == &init_task)
    break;
   if (!reaper->signal->is_child_subreaper)
    continue;
   thread = find_alive_thread(reaper);
   if (thread)
    return thread;
  }
 }

 return child_reaper;
}




static void reparent_leader(struct task_struct *father, struct task_struct *p,
    struct list_head *dead)
{
 if (unlikely(p->exit_state == EXIT_DEAD))
  return;


 p->exit_signal = SIGCHLD;


 if (!p->ptrace &&
     p->exit_state == EXIT_ZOMBIE && thread_group_empty(p)) {
  if (do_notify_parent(p, p->exit_signal)) {
   p->exit_state = EXIT_DEAD;
   list_add(&p->ptrace_entry, dead);
  }
 }

 kill_orphaned_pgrp(p, father);
}
static void forget_original_parent(struct task_struct *father,
     struct list_head *dead)
{
 struct task_struct *p, *t, *reaper;

 if (unlikely(!list_empty(&father->ptraced)))
  exit_ptrace(father, dead);


 reaper = find_child_reaper(father);
 if (list_empty(&father->children))
  return;

 reaper = find_new_reaper(father, reaper);
 list_for_each_entry(p, &father->children, sibling) {
  for_each_thread(p, t) {
   t->real_parent = reaper;
   BUG_ON((!t->ptrace) != (t->parent == father));
   if (likely(!t->ptrace))
    t->parent = t->real_parent;
   if (t->pdeath_signal)
    group_send_sig_info(t->pdeath_signal,
          SEND_SIG_NOINFO, t);
  }




  if (!same_thread_group(reaper, father))
   reparent_leader(father, p, dead);
 }
 list_splice_tail_init(&father->children, &reaper->children);
}





static void exit_notify(struct task_struct *tsk, int group_dead)
{
 bool autoreap;
 struct task_struct *p, *n;
 LIST_HEAD(dead);

 write_lock_irq(&tasklist_lock);
 forget_original_parent(tsk, &dead);

 if (group_dead)
  kill_orphaned_pgrp(tsk->group_leader, NULL);

 if (unlikely(tsk->ptrace)) {
  int sig = thread_group_leader(tsk) &&
    thread_group_empty(tsk) &&
    !ptrace_reparented(tsk) ?
   tsk->exit_signal : SIGCHLD;
  autoreap = do_notify_parent(tsk, sig);
 } else if (thread_group_leader(tsk)) {
  autoreap = thread_group_empty(tsk) &&
   do_notify_parent(tsk, tsk->exit_signal);
 } else {
  autoreap = true;
 }

 tsk->exit_state = autoreap ? EXIT_DEAD : EXIT_ZOMBIE;
 if (tsk->exit_state == EXIT_DEAD)
  list_add(&tsk->ptrace_entry, &dead);


 if (unlikely(tsk->signal->notify_count < 0))
  wake_up_process(tsk->signal->group_exit_task);
 write_unlock_irq(&tasklist_lock);

 list_for_each_entry_safe(p, n, &dead, ptrace_entry) {
  list_del_init(&p->ptrace_entry);
  release_task(p);
 }
}

static void check_stack_usage(void)
{
 static DEFINE_SPINLOCK(low_water_lock);
 static int lowest_to_date = THREAD_SIZE;
 unsigned long free;

 free = stack_not_used(current);

 if (free >= lowest_to_date)
  return;

 spin_lock(&low_water_lock);
 if (free < lowest_to_date) {
  pr_warn("%s (%d) used greatest stack depth: %lu bytes left\n",
   current->comm, task_pid_nr(current), free);
  lowest_to_date = free;
 }
 spin_unlock(&low_water_lock);
}
static inline void check_stack_usage(void) {}

void do_exit(long code)
{
 struct task_struct *tsk = current;
 int group_dead;
 TASKS_RCU(int tasks_rcu_i);

 profile_task_exit(tsk);
 kcov_task_exit(tsk);

 WARN_ON(blk_needs_flush_plug(tsk));

 if (unlikely(in_interrupt()))
  panic("Aiee, killing interrupt handler!");
 if (unlikely(!tsk->pid))
  panic("Attempted to kill the idle task!");
 set_fs(USER_DS);

 ptrace_event(PTRACE_EVENT_EXIT, code);

 validate_creds_for_do_exit(tsk);





 if (unlikely(tsk->flags & PF_EXITING)) {
  pr_alert("Fixing recursive fault but reboot is needed!\n");
  tsk->flags |= PF_EXITPIDONE;
  set_current_state(TASK_UNINTERRUPTIBLE);
  schedule();
 }

 exit_signals(tsk);




 smp_mb();
 raw_spin_unlock_wait(&tsk->pi_lock);

 if (unlikely(in_atomic())) {
  pr_info("note: %s[%d] exited with preempt_count %d\n",
   current->comm, task_pid_nr(current),
   preempt_count());
  preempt_count_set(PREEMPT_ENABLED);
 }


 if (tsk->mm)
  sync_mm_rss(tsk->mm);
 acct_update_integrals(tsk);
 group_dead = atomic_dec_and_test(&tsk->signal->live);
 if (group_dead) {
  hrtimer_cancel(&tsk->signal->real_timer);
  exit_itimers(tsk->signal);
  if (tsk->mm)
   setmax_mm_hiwater_rss(&tsk->signal->maxrss, tsk->mm);
 }
 acct_collect(code, group_dead);
 if (group_dead)
  tty_audit_exit();
 audit_free(tsk);

 tsk->exit_code = code;
 taskstats_exit(tsk, group_dead);

 exit_mm(tsk);

 if (group_dead)
  acct_process();
 trace_sched_process_exit(tsk);

 exit_sem(tsk);
 exit_shm(tsk);
 exit_files(tsk);
 exit_fs(tsk);
 if (group_dead)
  disassociate_ctty(1);
 exit_task_namespaces(tsk);
 exit_task_work(tsk);
 exit_thread(tsk);







 perf_event_exit_task(tsk);

 cgroup_exit(tsk);




 flush_ptrace_hw_breakpoint(tsk);

 TASKS_RCU(preempt_disable());
 TASKS_RCU(tasks_rcu_i = __srcu_read_lock(&tasks_rcu_exit_srcu));
 TASKS_RCU(preempt_enable());
 exit_notify(tsk, group_dead);
 proc_exit_connector(tsk);
 task_lock(tsk);
 mpol_put(tsk->mempolicy);
 tsk->mempolicy = NULL;
 task_unlock(tsk);
 if (unlikely(current->pi_state_cache))
  kfree(current->pi_state_cache);



 debug_check_no_locks_held();





 tsk->flags |= PF_EXITPIDONE;

 if (tsk->io_context)
  exit_io_context(tsk);

 if (tsk->splice_pipe)
  free_pipe_info(tsk->splice_pipe);

 if (tsk->task_frag.page)
  put_page(tsk->task_frag.page);

 validate_creds_for_do_exit(tsk);

 check_stack_usage();
 preempt_disable();
 if (tsk->nr_dirtied)
  __this_cpu_add(dirty_throttle_leaks, tsk->nr_dirtied);
 exit_rcu();
 TASKS_RCU(__srcu_read_unlock(&tasks_rcu_exit_srcu, tasks_rcu_i));
 smp_mb();
 raw_spin_unlock_wait(&tsk->pi_lock);


 tsk->state = TASK_DEAD;
 tsk->flags |= PF_NOFREEZE;
 schedule();
 BUG();

 for (;;)
  cpu_relax();
}
EXPORT_SYMBOL_GPL(do_exit);

void complete_and_exit(struct completion *comp, long code)
{
 if (comp)
  complete(comp);

 do_exit(code);
}
EXPORT_SYMBOL(complete_and_exit);

SYSCALL_DEFINE1(exit, int, error_code)
{
 do_exit((error_code&0xff)<<8);
}





void
do_group_exit(int exit_code)
{
 struct signal_struct *sig = current->signal;

 BUG_ON(exit_code & 0x80);

 if (signal_group_exit(sig))
  exit_code = sig->group_exit_code;
 else if (!thread_group_empty(current)) {
  struct sighand_struct *const sighand = current->sighand;

  spin_lock_irq(&sighand->siglock);
  if (signal_group_exit(sig))

   exit_code = sig->group_exit_code;
  else {
   sig->group_exit_code = exit_code;
   sig->flags = SIGNAL_GROUP_EXIT;
   zap_other_threads(current);
  }
  spin_unlock_irq(&sighand->siglock);
 }

 do_exit(exit_code);

}






SYSCALL_DEFINE1(exit_group, int, error_code)
{
 do_group_exit((error_code & 0xff) << 8);

 return 0;
}

struct wait_opts {
 enum pid_type wo_type;
 int wo_flags;
 struct pid *wo_pid;

 struct siginfo __user *wo_info;
 int __user *wo_stat;
 struct rusage __user *wo_rusage;

 wait_queue_t child_wait;
 int notask_error;
};

static inline
struct pid *task_pid_type(struct task_struct *task, enum pid_type type)
{
 if (type != PIDTYPE_PID)
  task = task->group_leader;
 return task->pids[type].pid;
}

static int eligible_pid(struct wait_opts *wo, struct task_struct *p)
{
 return wo->wo_type == PIDTYPE_MAX ||
  task_pid_type(p, wo->wo_type) == wo->wo_pid;
}

static int
eligible_child(struct wait_opts *wo, bool ptrace, struct task_struct *p)
{
 if (!eligible_pid(wo, p))
  return 0;





 if (ptrace || (wo->wo_flags & __WALL))
  return 1;
 if ((p->exit_signal != SIGCHLD) ^ !!(wo->wo_flags & __WCLONE))
  return 0;

 return 1;
}

static int wait_noreap_copyout(struct wait_opts *wo, struct task_struct *p,
    pid_t pid, uid_t uid, int why, int status)
{
 struct siginfo __user *infop;
 int retval = wo->wo_rusage
  ? getrusage(p, RUSAGE_BOTH, wo->wo_rusage) : 0;

 put_task_struct(p);
 infop = wo->wo_info;
 if (infop) {
  if (!retval)
   retval = put_user(SIGCHLD, &infop->si_signo);
  if (!retval)
   retval = put_user(0, &infop->si_errno);
  if (!retval)
   retval = put_user((short)why, &infop->si_code);
  if (!retval)
   retval = put_user(pid, &infop->si_pid);
  if (!retval)
   retval = put_user(uid, &infop->si_uid);
  if (!retval)
   retval = put_user(status, &infop->si_status);
 }
 if (!retval)
  retval = pid;
 return retval;
}







static int wait_task_zombie(struct wait_opts *wo, struct task_struct *p)
{
 int state, retval, status;
 pid_t pid = task_pid_vnr(p);
 uid_t uid = from_kuid_munged(current_user_ns(), task_uid(p));
 struct siginfo __user *infop;

 if (!likely(wo->wo_flags & WEXITED))
  return 0;

 if (unlikely(wo->wo_flags & WNOWAIT)) {
  int exit_code = p->exit_code;
  int why;

  get_task_struct(p);
  read_unlock(&tasklist_lock);
  sched_annotate_sleep();

  if ((exit_code & 0x7f) == 0) {
   why = CLD_EXITED;
   status = exit_code >> 8;
  } else {
   why = (exit_code & 0x80) ? CLD_DUMPED : CLD_KILLED;
   status = exit_code & 0x7f;
  }
  return wait_noreap_copyout(wo, p, pid, uid, why, status);
 }



 state = (ptrace_reparented(p) && thread_group_leader(p)) ?
  EXIT_TRACE : EXIT_DEAD;
 if (cmpxchg(&p->exit_state, EXIT_ZOMBIE, state) != EXIT_ZOMBIE)
  return 0;



 read_unlock(&tasklist_lock);
 sched_annotate_sleep();




 if (state == EXIT_DEAD && thread_group_leader(p)) {
  struct signal_struct *sig = p->signal;
  struct signal_struct *psig = current->signal;
  unsigned long maxrss;
  cputime_t tgutime, tgstime;
  thread_group_cputime_adjusted(p, &tgutime, &tgstime);
  spin_lock_irq(&current->sighand->siglock);
  write_seqlock(&psig->stats_lock);
  psig->cutime += tgutime + sig->cutime;
  psig->cstime += tgstime + sig->cstime;
  psig->cgtime += task_gtime(p) + sig->gtime + sig->cgtime;
  psig->cmin_flt +=
   p->min_flt + sig->min_flt + sig->cmin_flt;
  psig->cmaj_flt +=
   p->maj_flt + sig->maj_flt + sig->cmaj_flt;
  psig->cnvcsw +=
   p->nvcsw + sig->nvcsw + sig->cnvcsw;
  psig->cnivcsw +=
   p->nivcsw + sig->nivcsw + sig->cnivcsw;
  psig->cinblock +=
   task_io_get_inblock(p) +
   sig->inblock + sig->cinblock;
  psig->coublock +=
   task_io_get_oublock(p) +
   sig->oublock + sig->coublock;
  maxrss = max(sig->maxrss, sig->cmaxrss);
  if (psig->cmaxrss < maxrss)
   psig->cmaxrss = maxrss;
  task_io_accounting_add(&psig->ioac, &p->ioac);
  task_io_accounting_add(&psig->ioac, &sig->ioac);
  write_sequnlock(&psig->stats_lock);
  spin_unlock_irq(&current->sighand->siglock);
 }

 retval = wo->wo_rusage
  ? getrusage(p, RUSAGE_BOTH, wo->wo_rusage) : 0;
 status = (p->signal->flags & SIGNAL_GROUP_EXIT)
  ? p->signal->group_exit_code : p->exit_code;
 if (!retval && wo->wo_stat)
  retval = put_user(status, wo->wo_stat);

 infop = wo->wo_info;
 if (!retval && infop)
  retval = put_user(SIGCHLD, &infop->si_signo);
 if (!retval && infop)
  retval = put_user(0, &infop->si_errno);
 if (!retval && infop) {
  int why;

  if ((status & 0x7f) == 0) {
   why = CLD_EXITED;
   status >>= 8;
  } else {
   why = (status & 0x80) ? CLD_DUMPED : CLD_KILLED;
   status &= 0x7f;
  }
  retval = put_user((short)why, &infop->si_code);
  if (!retval)
   retval = put_user(status, &infop->si_status);
 }
 if (!retval && infop)
  retval = put_user(pid, &infop->si_pid);
 if (!retval && infop)
  retval = put_user(uid, &infop->si_uid);
 if (!retval)
  retval = pid;

 if (state == EXIT_TRACE) {
  write_lock_irq(&tasklist_lock);

  ptrace_unlink(p);


  state = EXIT_ZOMBIE;
  if (do_notify_parent(p, p->exit_signal))
   state = EXIT_DEAD;
  p->exit_state = state;
  write_unlock_irq(&tasklist_lock);
 }
 if (state == EXIT_DEAD)
  release_task(p);

 return retval;
}

static int *task_stopped_code(struct task_struct *p, bool ptrace)
{
 if (ptrace) {
  if (task_is_traced(p) && !(p->jobctl & JOBCTL_LISTENING))
   return &p->exit_code;
 } else {
  if (p->signal->flags & SIGNAL_STOP_STOPPED)
   return &p->signal->group_exit_code;
 }
 return NULL;
}
static int wait_task_stopped(struct wait_opts *wo,
    int ptrace, struct task_struct *p)
{
 struct siginfo __user *infop;
 int retval, exit_code, *p_code, why;
 uid_t uid = 0;
 pid_t pid;




 if (!ptrace && !(wo->wo_flags & WUNTRACED))
  return 0;

 if (!task_stopped_code(p, ptrace))
  return 0;

 exit_code = 0;
 spin_lock_irq(&p->sighand->siglock);

 p_code = task_stopped_code(p, ptrace);
 if (unlikely(!p_code))
  goto unlock_sig;

 exit_code = *p_code;
 if (!exit_code)
  goto unlock_sig;

 if (!unlikely(wo->wo_flags & WNOWAIT))
  *p_code = 0;

 uid = from_kuid_munged(current_user_ns(), task_uid(p));
unlock_sig:
 spin_unlock_irq(&p->sighand->siglock);
 if (!exit_code)
  return 0;
 get_task_struct(p);
 pid = task_pid_vnr(p);
 why = ptrace ? CLD_TRAPPED : CLD_STOPPED;
 read_unlock(&tasklist_lock);
 sched_annotate_sleep();

 if (unlikely(wo->wo_flags & WNOWAIT))
  return wait_noreap_copyout(wo, p, pid, uid, why, exit_code);

 retval = wo->wo_rusage
  ? getrusage(p, RUSAGE_BOTH, wo->wo_rusage) : 0;
 if (!retval && wo->wo_stat)
  retval = put_user((exit_code << 8) | 0x7f, wo->wo_stat);

 infop = wo->wo_info;
 if (!retval && infop)
  retval = put_user(SIGCHLD, &infop->si_signo);
 if (!retval && infop)
  retval = put_user(0, &infop->si_errno);
 if (!retval && infop)
  retval = put_user((short)why, &infop->si_code);
 if (!retval && infop)
  retval = put_user(exit_code, &infop->si_status);
 if (!retval && infop)
  retval = put_user(pid, &infop->si_pid);
 if (!retval && infop)
  retval = put_user(uid, &infop->si_uid);
 if (!retval)
  retval = pid;
 put_task_struct(p);

 BUG_ON(!retval);
 return retval;
}







static int wait_task_continued(struct wait_opts *wo, struct task_struct *p)
{
 int retval;
 pid_t pid;
 uid_t uid;

 if (!unlikely(wo->wo_flags & WCONTINUED))
  return 0;

 if (!(p->signal->flags & SIGNAL_STOP_CONTINUED))
  return 0;

 spin_lock_irq(&p->sighand->siglock);

 if (!(p->signal->flags & SIGNAL_STOP_CONTINUED)) {
  spin_unlock_irq(&p->sighand->siglock);
  return 0;
 }
 if (!unlikely(wo->wo_flags & WNOWAIT))
  p->signal->flags &= ~SIGNAL_STOP_CONTINUED;
 uid = from_kuid_munged(current_user_ns(), task_uid(p));
 spin_unlock_irq(&p->sighand->siglock);

 pid = task_pid_vnr(p);
 get_task_struct(p);
 read_unlock(&tasklist_lock);
 sched_annotate_sleep();

 if (!wo->wo_info) {
  retval = wo->wo_rusage
   ? getrusage(p, RUSAGE_BOTH, wo->wo_rusage) : 0;
  put_task_struct(p);
  if (!retval && wo->wo_stat)
   retval = put_user(0xffff, wo->wo_stat);
  if (!retval)
   retval = pid;
 } else {
  retval = wait_noreap_copyout(wo, p, pid, uid,
          CLD_CONTINUED, SIGCONT);
  BUG_ON(retval == 0);
 }

 return retval;
}
static int wait_consider_task(struct wait_opts *wo, int ptrace,
    struct task_struct *p)
{





 int exit_state = ACCESS_ONCE(p->exit_state);
 int ret;

 if (unlikely(exit_state == EXIT_DEAD))
  return 0;

 ret = eligible_child(wo, ptrace, p);
 if (!ret)
  return ret;

 ret = security_task_wait(p);
 if (unlikely(ret < 0)) {







  if (wo->notask_error)
   wo->notask_error = ret;
  return 0;
 }

 if (unlikely(exit_state == EXIT_TRACE)) {




  if (likely(!ptrace))
   wo->notask_error = 0;
  return 0;
 }

 if (likely(!ptrace) && unlikely(p->ptrace)) {
  if (!ptrace_reparented(p))
   ptrace = 1;
 }


 if (exit_state == EXIT_ZOMBIE) {

  if (!delay_group_leader(p)) {





   if (unlikely(ptrace) || likely(!p->ptrace))
    return wait_task_zombie(wo, p);
  }
  if (likely(!ptrace) || (wo->wo_flags & (WCONTINUED | WEXITED)))
   wo->notask_error = 0;
 } else {




  wo->notask_error = 0;
 }





 ret = wait_task_stopped(wo, ptrace, p);
 if (ret)
  return ret;






 return wait_task_continued(wo, p);
}
static int do_wait_thread(struct wait_opts *wo, struct task_struct *tsk)
{
 struct task_struct *p;

 list_for_each_entry(p, &tsk->children, sibling) {
  int ret = wait_consider_task(wo, 0, p);

  if (ret)
   return ret;
 }

 return 0;
}

static int ptrace_do_wait(struct wait_opts *wo, struct task_struct *tsk)
{
 struct task_struct *p;

 list_for_each_entry(p, &tsk->ptraced, ptrace_entry) {
  int ret = wait_consider_task(wo, 1, p);

  if (ret)
   return ret;
 }

 return 0;
}

static int child_wait_callback(wait_queue_t *wait, unsigned mode,
    int sync, void *key)
{
 struct wait_opts *wo = container_of(wait, struct wait_opts,
      child_wait);
 struct task_struct *p = key;

 if (!eligible_pid(wo, p))
  return 0;

 if ((wo->wo_flags & __WNOTHREAD) && wait->private != p->parent)
  return 0;

 return default_wake_function(wait, mode, sync, key);
}

void __wake_up_parent(struct task_struct *p, struct task_struct *parent)
{
 __wake_up_sync_key(&parent->signal->wait_chldexit,
    TASK_INTERRUPTIBLE, 1, p);
}

static long do_wait(struct wait_opts *wo)
{
 struct task_struct *tsk;
 int retval;

 trace_sched_process_wait(wo->wo_pid);

 init_waitqueue_func_entry(&wo->child_wait, child_wait_callback);
 wo->child_wait.private = current;
 add_wait_queue(&current->signal->wait_chldexit, &wo->child_wait);
repeat:






 wo->notask_error = -ECHILD;
 if ((wo->wo_type < PIDTYPE_MAX) &&
    (!wo->wo_pid || hlist_empty(&wo->wo_pid->tasks[wo->wo_type])))
  goto notask;

 set_current_state(TASK_INTERRUPTIBLE);
 read_lock(&tasklist_lock);
 tsk = current;
 do {
  retval = do_wait_thread(wo, tsk);
  if (retval)
   goto end;

  retval = ptrace_do_wait(wo, tsk);
  if (retval)
   goto end;

  if (wo->wo_flags & __WNOTHREAD)
   break;
 } while_each_thread(current, tsk);
 read_unlock(&tasklist_lock);

notask:
 retval = wo->notask_error;
 if (!retval && !(wo->wo_flags & WNOHANG)) {
  retval = -ERESTARTSYS;
  if (!signal_pending(current)) {
   schedule();
   goto repeat;
  }
 }
end:
 __set_current_state(TASK_RUNNING);
 remove_wait_queue(&current->signal->wait_chldexit, &wo->child_wait);
 return retval;
}

SYSCALL_DEFINE5(waitid, int, which, pid_t, upid, struct siginfo __user *,
  infop, int, options, struct rusage __user *, ru)
{
 struct wait_opts wo;
 struct pid *pid = NULL;
 enum pid_type type;
 long ret;

 if (options & ~(WNOHANG|WNOWAIT|WEXITED|WSTOPPED|WCONTINUED|
   __WNOTHREAD|__WCLONE|__WALL))
  return -EINVAL;
 if (!(options & (WEXITED|WSTOPPED|WCONTINUED)))
  return -EINVAL;

 switch (which) {
 case P_ALL:
  type = PIDTYPE_MAX;
  break;
 case P_PID:
  type = PIDTYPE_PID;
  if (upid <= 0)
   return -EINVAL;
  break;
 case P_PGID:
  type = PIDTYPE_PGID;
  if (upid <= 0)
   return -EINVAL;
  break;
 default:
  return -EINVAL;
 }

 if (type < PIDTYPE_MAX)
  pid = find_get_pid(upid);

 wo.wo_type = type;
 wo.wo_pid = pid;
 wo.wo_flags = options;
 wo.wo_info = infop;
 wo.wo_stat = NULL;
 wo.wo_rusage = ru;
 ret = do_wait(&wo);

 if (ret > 0) {
  ret = 0;
 } else if (infop) {





  if (!ret)
   ret = put_user(0, &infop->si_signo);
  if (!ret)
   ret = put_user(0, &infop->si_errno);
  if (!ret)
   ret = put_user(0, &infop->si_code);
  if (!ret)
   ret = put_user(0, &infop->si_pid);
  if (!ret)
   ret = put_user(0, &infop->si_uid);
  if (!ret)
   ret = put_user(0, &infop->si_status);
 }

 put_pid(pid);
 return ret;
}

SYSCALL_DEFINE4(wait4, pid_t, upid, int __user *, stat_addr,
  int, options, struct rusage __user *, ru)
{
 struct wait_opts wo;
 struct pid *pid = NULL;
 enum pid_type type;
 long ret;

 if (options & ~(WNOHANG|WUNTRACED|WCONTINUED|
   __WNOTHREAD|__WCLONE|__WALL))
  return -EINVAL;

 if (upid == -1)
  type = PIDTYPE_MAX;
 else if (upid < 0) {
  type = PIDTYPE_PGID;
  pid = find_get_pid(-upid);
 } else if (upid == 0) {
  type = PIDTYPE_PGID;
  pid = get_task_pid(current, PIDTYPE_PGID);
 } else {
  type = PIDTYPE_PID;
  pid = find_get_pid(upid);
 }

 wo.wo_type = type;
 wo.wo_pid = pid;
 wo.wo_flags = options | WEXITED;
 wo.wo_info = NULL;
 wo.wo_stat = stat_addr;
 wo.wo_rusage = ru;
 ret = do_wait(&wo);
 put_pid(pid);

 return ret;
}






SYSCALL_DEFINE3(waitpid, pid_t, pid, int __user *, stat_addr, int, options)
{
 return sys_wait4(pid, stat_addr, options, NULL);
}









DEFINE_MUTEX(text_mutex);

extern struct exception_table_entry __start___ex_table[];
extern struct exception_table_entry __stop___ex_table[];


u32 __initdata __visible main_extable_sort_needed = 1;


void __init sort_main_extable(void)
{
 if (main_extable_sort_needed && __stop___ex_table > __start___ex_table) {
  pr_notice("Sorting __ex_table...\n");
  sort_extable(__start___ex_table, __stop___ex_table);
 }
}


const struct exception_table_entry *search_exception_tables(unsigned long addr)
{
 const struct exception_table_entry *e;

 e = search_extable(__start___ex_table, __stop___ex_table-1, addr);
 if (!e)
  e = search_module_extables(addr);
 return e;
}

static inline int init_kernel_text(unsigned long addr)
{
 if (addr >= (unsigned long)_sinittext &&
     addr < (unsigned long)_einittext)
  return 1;
 return 0;
}

int core_kernel_text(unsigned long addr)
{
 if (addr >= (unsigned long)_stext &&
     addr < (unsigned long)_etext)
  return 1;

 if (system_state == SYSTEM_BOOTING &&
     init_kernel_text(addr))
  return 1;
 return 0;
}
int core_kernel_data(unsigned long addr)
{
 if (addr >= (unsigned long)_sdata &&
     addr < (unsigned long)_edata)
  return 1;
 return 0;
}

int __kernel_text_address(unsigned long addr)
{
 if (core_kernel_text(addr))
  return 1;
 if (is_module_text_address(addr))
  return 1;
 if (is_ftrace_trampoline(addr))
  return 1;
 if (init_kernel_text(addr))
  return 1;
 return 0;
}

int kernel_text_address(unsigned long addr)
{
 if (core_kernel_text(addr))
  return 1;
 if (is_module_text_address(addr))
  return 1;
 return is_ftrace_trampoline(addr);
}
int func_ptr_is_kernel_text(void *ptr)
{
 unsigned long addr;
 addr = (unsigned long) dereference_function_descriptor(ptr);
 if (core_kernel_text(addr))
  return 1;
 return is_module_text_address(addr);
}















unsigned long total_forks;
int nr_threads;

int max_threads;

DEFINE_PER_CPU(unsigned long, process_counts) = 0;

__cacheline_aligned DEFINE_RWLOCK(tasklist_lock);

int lockdep_tasklist_lock_is_held(void)
{
 return lockdep_is_held(&tasklist_lock);
}
EXPORT_SYMBOL_GPL(lockdep_tasklist_lock_is_held);

int nr_processes(void)
{
 int cpu;
 int total = 0;

 for_each_possible_cpu(cpu)
  total += per_cpu(process_counts, cpu);

 return total;
}

void __weak arch_release_task_struct(struct task_struct *tsk)
{
}

static struct kmem_cache *task_struct_cachep;

static inline struct task_struct *alloc_task_struct_node(int node)
{
 return kmem_cache_alloc_node(task_struct_cachep, GFP_KERNEL, node);
}

static inline void free_task_struct(struct task_struct *tsk)
{
 kmem_cache_free(task_struct_cachep, tsk);
}

void __weak arch_release_thread_stack(unsigned long *stack)
{
}






static unsigned long *alloc_thread_stack_node(struct task_struct *tsk,
        int node)
{
 struct page *page = alloc_kmem_pages_node(node, THREADINFO_GFP,
        THREAD_SIZE_ORDER);

 if (page)
  memcg_kmem_update_page_stat(page, MEMCG_KERNEL_STACK,
         1 << THREAD_SIZE_ORDER);

 return page ? page_address(page) : NULL;
}

static inline void free_thread_stack(unsigned long *stack)
{
 struct page *page = virt_to_page(stack);

 memcg_kmem_update_page_stat(page, MEMCG_KERNEL_STACK,
        -(1 << THREAD_SIZE_ORDER));
 __free_kmem_pages(page, THREAD_SIZE_ORDER);
}
static struct kmem_cache *thread_stack_cache;

static unsigned long *alloc_thread_stack_node(struct task_struct *tsk,
        int node)
{
 return kmem_cache_alloc_node(thread_stack_cache, THREADINFO_GFP, node);
}

static void free_thread_stack(unsigned long *stack)
{
 kmem_cache_free(thread_stack_cache, stack);
}

void thread_stack_cache_init(void)
{
 thread_stack_cache = kmem_cache_create("thread_stack", THREAD_SIZE,
           THREAD_SIZE, 0, NULL);
 BUG_ON(thread_stack_cache == NULL);
}


static struct kmem_cache *signal_cachep;


struct kmem_cache *sighand_cachep;


struct kmem_cache *files_cachep;


struct kmem_cache *fs_cachep;


struct kmem_cache *vm_area_cachep;


static struct kmem_cache *mm_cachep;

static void account_kernel_stack(unsigned long *stack, int account)
{
 struct zone *zone = page_zone(virt_to_page(stack));

 mod_zone_page_state(zone, NR_KERNEL_STACK, account);
}

void free_task(struct task_struct *tsk)
{
 account_kernel_stack(tsk->stack, -1);
 arch_release_thread_stack(tsk->stack);
 free_thread_stack(tsk->stack);
 rt_mutex_debug_task_free(tsk);
 ftrace_graph_exit_task(tsk);
 put_seccomp_filter(tsk);
 arch_release_task_struct(tsk);
 free_task_struct(tsk);
}
EXPORT_SYMBOL(free_task);

static inline void free_signal_struct(struct signal_struct *sig)
{
 taskstats_tgid_free(sig);
 sched_autogroup_exit(sig);
 kmem_cache_free(signal_cachep, sig);
}

static inline void put_signal_struct(struct signal_struct *sig)
{
 if (atomic_dec_and_test(&sig->sigcnt))
  free_signal_struct(sig);
}

void __put_task_struct(struct task_struct *tsk)
{
 WARN_ON(!tsk->exit_state);
 WARN_ON(atomic_read(&tsk->usage));
 WARN_ON(tsk == current);

 cgroup_free(tsk);
 task_numa_free(tsk);
 security_task_free(tsk);
 exit_creds(tsk);
 delayacct_tsk_free(tsk);
 put_signal_struct(tsk->signal);

 if (!profile_handoff_task(tsk))
  free_task(tsk);
}
EXPORT_SYMBOL_GPL(__put_task_struct);

void __init __weak arch_task_cache_init(void) { }




static void set_max_threads(unsigned int max_threads_suggested)
{
 u64 threads;





 if (fls64(totalram_pages) + fls64(PAGE_SIZE) > 64)
  threads = MAX_THREADS;
 else
  threads = div64_u64((u64) totalram_pages * (u64) PAGE_SIZE,
        (u64) THREAD_SIZE * 8UL);

 if (threads > max_threads_suggested)
  threads = max_threads_suggested;

 max_threads = clamp_t(u64, threads, MIN_THREADS, MAX_THREADS);
}


int arch_task_struct_size __read_mostly;

void __init fork_init(void)
{

 task_struct_cachep = kmem_cache_create("task_struct",
   arch_task_struct_size, ARCH_MIN_TASKALIGN,
   SLAB_PANIC|SLAB_NOTRACK|SLAB_ACCOUNT, NULL);


 arch_task_cache_init();

 set_max_threads(MAX_THREADS);

 init_task.signal->rlim[RLIMIT_NPROC].rlim_cur = max_threads/2;
 init_task.signal->rlim[RLIMIT_NPROC].rlim_max = max_threads/2;
 init_task.signal->rlim[RLIMIT_SIGPENDING] =
  init_task.signal->rlim[RLIMIT_NPROC];
}

int __weak arch_dup_task_struct(struct task_struct *dst,
            struct task_struct *src)
{
 *dst = *src;
 return 0;
}

void set_task_stack_end_magic(struct task_struct *tsk)
{
 unsigned long *stackend;

 stackend = end_of_stack(tsk);
 *stackend = STACK_END_MAGIC;
}

static struct task_struct *dup_task_struct(struct task_struct *orig, int node)
{
 struct task_struct *tsk;
 unsigned long *stack;
 int err;

 if (node == NUMA_NO_NODE)
  node = tsk_fork_get_node(orig);
 tsk = alloc_task_struct_node(node);
 if (!tsk)
  return NULL;

 stack = alloc_thread_stack_node(tsk, node);
 if (!stack)
  goto free_tsk;

 err = arch_dup_task_struct(tsk, orig);
 if (err)
  goto free_stack;

 tsk->stack = stack;






 tsk->seccomp.filter = NULL;

 setup_thread_stack(tsk, orig);
 clear_user_return_notifier(tsk);
 clear_tsk_need_resched(tsk);
 set_task_stack_end_magic(tsk);

 tsk->stack_canary = get_random_int();





 atomic_set(&tsk->usage, 2);
 tsk->btrace_seq = 0;
 tsk->splice_pipe = NULL;
 tsk->task_frag.page = NULL;
 tsk->wake_q.next = NULL;

 account_kernel_stack(stack, 1);

 kcov_task_init(tsk);

 return tsk;

free_stack:
 free_thread_stack(stack);
free_tsk:
 free_task_struct(tsk);
 return NULL;
}

static int dup_mmap(struct mm_struct *mm, struct mm_struct *oldmm)
{
 struct vm_area_struct *mpnt, *tmp, *prev, **pprev;
 struct rb_node **rb_link, *rb_parent;
 int retval;
 unsigned long charge;

 uprobe_start_dup_mmap();
 if (down_write_killable(&oldmm->mmap_sem)) {
  retval = -EINTR;
  goto fail_uprobe_end;
 }
 flush_cache_dup_mm(oldmm);
 uprobe_dup_mmap(oldmm, mm);



 down_write_nested(&mm->mmap_sem, SINGLE_DEPTH_NESTING);


 RCU_INIT_POINTER(mm->exe_file, get_mm_exe_file(oldmm));

 mm->total_vm = oldmm->total_vm;
 mm->data_vm = oldmm->data_vm;
 mm->exec_vm = oldmm->exec_vm;
 mm->stack_vm = oldmm->stack_vm;

 rb_link = &mm->mm_rb.rb_node;
 rb_parent = NULL;
 pprev = &mm->mmap;
 retval = ksm_fork(mm, oldmm);
 if (retval)
  goto out;
 retval = khugepaged_fork(mm, oldmm);
 if (retval)
  goto out;

 prev = NULL;
 for (mpnt = oldmm->mmap; mpnt; mpnt = mpnt->vm_next) {
  struct file *file;

  if (mpnt->vm_flags & VM_DONTCOPY) {
   vm_stat_account(mm, mpnt->vm_flags, -vma_pages(mpnt));
   continue;
  }
  charge = 0;
  if (mpnt->vm_flags & VM_ACCOUNT) {
   unsigned long len = vma_pages(mpnt);

   if (security_vm_enough_memory_mm(oldmm, len))
    goto fail_nomem;
   charge = len;
  }
  tmp = kmem_cache_alloc(vm_area_cachep, GFP_KERNEL);
  if (!tmp)
   goto fail_nomem;
  *tmp = *mpnt;
  INIT_LIST_HEAD(&tmp->anon_vma_chain);
  retval = vma_dup_policy(mpnt, tmp);
  if (retval)
   goto fail_nomem_policy;
  tmp->vm_mm = mm;
  if (anon_vma_fork(tmp, mpnt))
   goto fail_nomem_anon_vma_fork;
  tmp->vm_flags &=
   ~(VM_LOCKED|VM_LOCKONFAULT|VM_UFFD_MISSING|VM_UFFD_WP);
  tmp->vm_next = tmp->vm_prev = NULL;
  tmp->vm_userfaultfd_ctx = NULL_VM_UFFD_CTX;
  file = tmp->vm_file;
  if (file) {
   struct inode *inode = file_inode(file);
   struct address_space *mapping = file->f_mapping;

   get_file(file);
   if (tmp->vm_flags & VM_DENYWRITE)
    atomic_dec(&inode->i_writecount);
   i_mmap_lock_write(mapping);
   if (tmp->vm_flags & VM_SHARED)
    atomic_inc(&mapping->i_mmap_writable);
   flush_dcache_mmap_lock(mapping);

   vma_interval_tree_insert_after(tmp, mpnt,
     &mapping->i_mmap);
   flush_dcache_mmap_unlock(mapping);
   i_mmap_unlock_write(mapping);
  }






  if (is_vm_hugetlb_page(tmp))
   reset_vma_resv_huge_pages(tmp);




  *pprev = tmp;
  pprev = &tmp->vm_next;
  tmp->vm_prev = prev;
  prev = tmp;

  __vma_link_rb(mm, tmp, rb_link, rb_parent);
  rb_link = &tmp->vm_rb.rb_right;
  rb_parent = &tmp->vm_rb;

  mm->map_count++;
  retval = copy_page_range(mm, oldmm, mpnt);

  if (tmp->vm_ops && tmp->vm_ops->open)
   tmp->vm_ops->open(tmp);

  if (retval)
   goto out;
 }

 arch_dup_mmap(oldmm, mm);
 retval = 0;
out:
 up_write(&mm->mmap_sem);
 flush_tlb_mm(oldmm);
 up_write(&oldmm->mmap_sem);
fail_uprobe_end:
 uprobe_end_dup_mmap();
 return retval;
fail_nomem_anon_vma_fork:
 mpol_put(vma_policy(tmp));
fail_nomem_policy:
 kmem_cache_free(vm_area_cachep, tmp);
fail_nomem:
 retval = -ENOMEM;
 vm_unacct_memory(charge);
 goto out;
}

static inline int mm_alloc_pgd(struct mm_struct *mm)
{
 mm->pgd = pgd_alloc(mm);
 if (unlikely(!mm->pgd))
  return -ENOMEM;
 return 0;
}

static inline void mm_free_pgd(struct mm_struct *mm)
{
 pgd_free(mm, mm->pgd);
}
static int dup_mmap(struct mm_struct *mm, struct mm_struct *oldmm)
{
 down_write(&oldmm->mmap_sem);
 RCU_INIT_POINTER(mm->exe_file, get_mm_exe_file(oldmm));
 up_write(&oldmm->mmap_sem);
 return 0;
}

__cacheline_aligned_in_smp DEFINE_SPINLOCK(mmlist_lock);


static unsigned long default_dump_filter = MMF_DUMP_FILTER_DEFAULT;

static int __init coredump_filter_setup(char *s)
{
 default_dump_filter =
  (simple_strtoul(s, NULL, 0) << MMF_DUMP_FILTER_SHIFT) &
  MMF_DUMP_FILTER_MASK;
 return 1;
}

__setup("coredump_filter=", coredump_filter_setup);


static void mm_init_aio(struct mm_struct *mm)
{
 spin_lock_init(&mm->ioctx_lock);
 mm->ioctx_table = NULL;
}

static void mm_init_owner(struct mm_struct *mm, struct task_struct *p)
{
 mm->owner = p;
}

static struct mm_struct *mm_init(struct mm_struct *mm, struct task_struct *p)
{
 mm->mmap = NULL;
 mm->mm_rb = RB_ROOT;
 mm->vmacache_seqnum = 0;
 atomic_set(&mm->mm_users, 1);
 atomic_set(&mm->mm_count, 1);
 init_rwsem(&mm->mmap_sem);
 INIT_LIST_HEAD(&mm->mmlist);
 mm->core_state = NULL;
 atomic_long_set(&mm->nr_ptes, 0);
 mm_nr_pmds_init(mm);
 mm->map_count = 0;
 mm->locked_vm = 0;
 mm->pinned_vm = 0;
 memset(&mm->rss_stat, 0, sizeof(mm->rss_stat));
 spin_lock_init(&mm->page_table_lock);
 mm_init_cpumask(mm);
 mm_init_aio(mm);
 mm_init_owner(mm, p);
 mmu_notifier_mm_init(mm);
 clear_tlb_flush_pending(mm);
 mm->pmd_huge_pte = NULL;

 if (current->mm) {
  mm->flags = current->mm->flags & MMF_INIT_MASK;
  mm->def_flags = current->mm->def_flags & VM_INIT_DEF_MASK;
 } else {
  mm->flags = default_dump_filter;
  mm->def_flags = 0;
 }

 if (mm_alloc_pgd(mm))
  goto fail_nopgd;

 if (init_new_context(p, mm))
  goto fail_nocontext;

 return mm;

fail_nocontext:
 mm_free_pgd(mm);
fail_nopgd:
 free_mm(mm);
 return NULL;
}

static void check_mm(struct mm_struct *mm)
{
 int i;

 for (i = 0; i < NR_MM_COUNTERS; i++) {
  long x = atomic_long_read(&mm->rss_stat.count[i]);

  if (unlikely(x))
   printk(KERN_ALERT "BUG: Bad rss-counter state "
       "mm:%p idx:%d val:%ld\n", mm, i, x);
 }

 if (atomic_long_read(&mm->nr_ptes))
  pr_alert("BUG: non-zero nr_ptes on freeing mm: %ld\n",
    atomic_long_read(&mm->nr_ptes));
 if (mm_nr_pmds(mm))
  pr_alert("BUG: non-zero nr_pmds on freeing mm: %ld\n",
    mm_nr_pmds(mm));

 VM_BUG_ON_MM(mm->pmd_huge_pte, mm);
}




struct mm_struct *mm_alloc(void)
{
 struct mm_struct *mm;

 mm = allocate_mm();
 if (!mm)
  return NULL;

 memset(mm, 0, sizeof(*mm));
 return mm_init(mm, current);
}






void __mmdrop(struct mm_struct *mm)
{
 BUG_ON(mm == &init_mm);
 mm_free_pgd(mm);
 destroy_context(mm);
 mmu_notifier_mm_destroy(mm);
 check_mm(mm);
 free_mm(mm);
}
EXPORT_SYMBOL_GPL(__mmdrop);

static inline void __mmput(struct mm_struct *mm)
{
 VM_BUG_ON(atomic_read(&mm->mm_users));

 uprobe_clear_state(mm);
 exit_aio(mm);
 ksm_exit(mm);
 khugepaged_exit(mm);
 exit_mmap(mm);
 set_mm_exe_file(mm, NULL);
 if (!list_empty(&mm->mmlist)) {
  spin_lock(&mmlist_lock);
  list_del(&mm->mmlist);
  spin_unlock(&mmlist_lock);
 }
 if (mm->binfmt)
  module_put(mm->binfmt->module);
 mmdrop(mm);
}




void mmput(struct mm_struct *mm)
{
 might_sleep();

 if (atomic_dec_and_test(&mm->mm_users))
  __mmput(mm);
}
EXPORT_SYMBOL_GPL(mmput);

static void mmput_async_fn(struct work_struct *work)
{
 struct mm_struct *mm = container_of(work, struct mm_struct, async_put_work);
 __mmput(mm);
}

void mmput_async(struct mm_struct *mm)
{
 if (atomic_dec_and_test(&mm->mm_users)) {
  INIT_WORK(&mm->async_put_work, mmput_async_fn);
  schedule_work(&mm->async_put_work);
 }
}
void set_mm_exe_file(struct mm_struct *mm, struct file *new_exe_file)
{
 struct file *old_exe_file;






 old_exe_file = rcu_dereference_raw(mm->exe_file);

 if (new_exe_file)
  get_file(new_exe_file);
 rcu_assign_pointer(mm->exe_file, new_exe_file);
 if (old_exe_file)
  fput(old_exe_file);
}







struct file *get_mm_exe_file(struct mm_struct *mm)
{
 struct file *exe_file;

 rcu_read_lock();
 exe_file = rcu_dereference(mm->exe_file);
 if (exe_file && !get_file_rcu(exe_file))
  exe_file = NULL;
 rcu_read_unlock();
 return exe_file;
}
EXPORT_SYMBOL(get_mm_exe_file);
struct mm_struct *get_task_mm(struct task_struct *task)
{
 struct mm_struct *mm;

 task_lock(task);
 mm = task->mm;
 if (mm) {
  if (task->flags & PF_KTHREAD)
   mm = NULL;
  else
   atomic_inc(&mm->mm_users);
 }
 task_unlock(task);
 return mm;
}
EXPORT_SYMBOL_GPL(get_task_mm);

struct mm_struct *mm_access(struct task_struct *task, unsigned int mode)
{
 struct mm_struct *mm;
 int err;

 err = mutex_lock_killable(&task->signal->cred_guard_mutex);
 if (err)
  return ERR_PTR(err);

 mm = get_task_mm(task);
 if (mm && mm != current->mm &&
   !ptrace_may_access(task, mode)) {
  mmput(mm);
  mm = ERR_PTR(-EACCES);
 }
 mutex_unlock(&task->signal->cred_guard_mutex);

 return mm;
}

static void complete_vfork_done(struct task_struct *tsk)
{
 struct completion *vfork;

 task_lock(tsk);
 vfork = tsk->vfork_done;
 if (likely(vfork)) {
  tsk->vfork_done = NULL;
  complete(vfork);
 }
 task_unlock(tsk);
}

static int wait_for_vfork_done(struct task_struct *child,
    struct completion *vfork)
{
 int killed;

 freezer_do_not_count();
 killed = wait_for_completion_killable(vfork);
 freezer_count();

 if (killed) {
  task_lock(child);
  child->vfork_done = NULL;
  task_unlock(child);
 }

 put_task_struct(child);
 return killed;
}
void mm_release(struct task_struct *tsk, struct mm_struct *mm)
{

 if (unlikely(tsk->robust_list)) {
  exit_robust_list(tsk);
  tsk->robust_list = NULL;
 }
 if (unlikely(tsk->compat_robust_list)) {
  compat_exit_robust_list(tsk);
  tsk->compat_robust_list = NULL;
 }
 if (unlikely(!list_empty(&tsk->pi_state_list)))
  exit_pi_state_list(tsk);

 uprobe_free_utask(tsk);


 deactivate_mm(tsk, mm);
 if (tsk->clear_child_tid) {
  if (!(tsk->flags & PF_SIGNALED) &&
      atomic_read(&mm->mm_users) > 1) {




   put_user(0, tsk->clear_child_tid);
   sys_futex(tsk->clear_child_tid, FUTEX_WAKE,
     1, NULL, NULL, 0);
  }
  tsk->clear_child_tid = NULL;
 }





 if (tsk->vfork_done)
  complete_vfork_done(tsk);
}





static struct mm_struct *dup_mm(struct task_struct *tsk)
{
 struct mm_struct *mm, *oldmm = current->mm;
 int err;

 mm = allocate_mm();
 if (!mm)
  goto fail_nomem;

 memcpy(mm, oldmm, sizeof(*mm));

 if (!mm_init(mm, tsk))
  goto fail_nomem;

 err = dup_mmap(mm, oldmm);
 if (err)
  goto free_pt;

 mm->hiwater_rss = get_mm_rss(mm);
 mm->hiwater_vm = mm->total_vm;

 if (mm->binfmt && !try_module_get(mm->binfmt->module))
  goto free_pt;

 return mm;

free_pt:

 mm->binfmt = NULL;
 mmput(mm);

fail_nomem:
 return NULL;
}

static int copy_mm(unsigned long clone_flags, struct task_struct *tsk)
{
 struct mm_struct *mm, *oldmm;
 int retval;

 tsk->min_flt = tsk->maj_flt = 0;
 tsk->nvcsw = tsk->nivcsw = 0;
 tsk->last_switch_count = tsk->nvcsw + tsk->nivcsw;

 tsk->mm = NULL;
 tsk->active_mm = NULL;






 oldmm = current->mm;
 if (!oldmm)
  return 0;


 vmacache_flush(tsk);

 if (clone_flags & CLONE_VM) {
  atomic_inc(&oldmm->mm_users);
  mm = oldmm;
  goto good_mm;
 }

 retval = -ENOMEM;
 mm = dup_mm(tsk);
 if (!mm)
  goto fail_nomem;

good_mm:
 tsk->mm = mm;
 tsk->active_mm = mm;
 return 0;

fail_nomem:
 return retval;
}

static int copy_fs(unsigned long clone_flags, struct task_struct *tsk)
{
 struct fs_struct *fs = current->fs;
 if (clone_flags & CLONE_FS) {

  spin_lock(&fs->lock);
  if (fs->in_exec) {
   spin_unlock(&fs->lock);
   return -EAGAIN;
  }
  fs->users++;
  spin_unlock(&fs->lock);
  return 0;
 }
 tsk->fs = copy_fs_struct(fs);
 if (!tsk->fs)
  return -ENOMEM;
 return 0;
}

static int copy_files(unsigned long clone_flags, struct task_struct *tsk)
{
 struct files_struct *oldf, *newf;
 int error = 0;




 oldf = current->files;
 if (!oldf)
  goto out;

 if (clone_flags & CLONE_FILES) {
  atomic_inc(&oldf->count);
  goto out;
 }

 newf = dup_fd(oldf, &error);
 if (!newf)
  goto out;

 tsk->files = newf;
 error = 0;
out:
 return error;
}

static int copy_io(unsigned long clone_flags, struct task_struct *tsk)
{
 struct io_context *ioc = current->io_context;
 struct io_context *new_ioc;

 if (!ioc)
  return 0;



 if (clone_flags & CLONE_IO) {
  ioc_task_link(ioc);
  tsk->io_context = ioc;
 } else if (ioprio_valid(ioc->ioprio)) {
  new_ioc = get_task_io_context(tsk, GFP_KERNEL, NUMA_NO_NODE);
  if (unlikely(!new_ioc))
   return -ENOMEM;

  new_ioc->ioprio = ioc->ioprio;
  put_io_context(new_ioc);
 }
 return 0;
}

static int copy_sighand(unsigned long clone_flags, struct task_struct *tsk)
{
 struct sighand_struct *sig;

 if (clone_flags & CLONE_SIGHAND) {
  atomic_inc(&current->sighand->count);
  return 0;
 }
 sig = kmem_cache_alloc(sighand_cachep, GFP_KERNEL);
 rcu_assign_pointer(tsk->sighand, sig);
 if (!sig)
  return -ENOMEM;

 atomic_set(&sig->count, 1);
 memcpy(sig->action, current->sighand->action, sizeof(sig->action));
 return 0;
}

void __cleanup_sighand(struct sighand_struct *sighand)
{
 if (atomic_dec_and_test(&sighand->count)) {
  signalfd_cleanup(sighand);




  kmem_cache_free(sighand_cachep, sighand);
 }
}




static void posix_cpu_timers_init_group(struct signal_struct *sig)
{
 unsigned long cpu_limit;

 cpu_limit = READ_ONCE(sig->rlim[RLIMIT_CPU].rlim_cur);
 if (cpu_limit != RLIM_INFINITY) {
  sig->cputime_expires.prof_exp = secs_to_cputime(cpu_limit);
  sig->cputimer.running = true;
 }


 INIT_LIST_HEAD(&sig->cpu_timers[0]);
 INIT_LIST_HEAD(&sig->cpu_timers[1]);
 INIT_LIST_HEAD(&sig->cpu_timers[2]);
}

static int copy_signal(unsigned long clone_flags, struct task_struct *tsk)
{
 struct signal_struct *sig;

 if (clone_flags & CLONE_THREAD)
  return 0;

 sig = kmem_cache_zalloc(signal_cachep, GFP_KERNEL);
 tsk->signal = sig;
 if (!sig)
  return -ENOMEM;

 sig->nr_threads = 1;
 atomic_set(&sig->live, 1);
 atomic_set(&sig->sigcnt, 1);


 sig->thread_head = (struct list_head)LIST_HEAD_INIT(tsk->thread_node);
 tsk->thread_node = (struct list_head)LIST_HEAD_INIT(sig->thread_head);

 init_waitqueue_head(&sig->wait_chldexit);
 sig->curr_target = tsk;
 init_sigpending(&sig->shared_pending);
 INIT_LIST_HEAD(&sig->posix_timers);
 seqlock_init(&sig->stats_lock);
 prev_cputime_init(&sig->prev_cputime);

 hrtimer_init(&sig->real_timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
 sig->real_timer.function = it_real_fn;

 task_lock(current->group_leader);
 memcpy(sig->rlim, current->signal->rlim, sizeof sig->rlim);
 task_unlock(current->group_leader);

 posix_cpu_timers_init_group(sig);

 tty_audit_fork(sig);
 sched_autogroup_fork(sig);

 sig->oom_score_adj = current->signal->oom_score_adj;
 sig->oom_score_adj_min = current->signal->oom_score_adj_min;

 sig->has_child_subreaper = current->signal->has_child_subreaper ||
       current->signal->is_child_subreaper;

 mutex_init(&sig->cred_guard_mutex);

 return 0;
}

static void copy_seccomp(struct task_struct *p)
{






 assert_spin_locked(&current->sighand->siglock);


 get_seccomp_filter(current);
 p->seccomp = current->seccomp;






 if (task_no_new_privs(current))
  task_set_no_new_privs(p);






 if (p->seccomp.mode != SECCOMP_MODE_DISABLED)
  set_tsk_thread_flag(p, TIF_SECCOMP);
}

SYSCALL_DEFINE1(set_tid_address, int __user *, tidptr)
{
 current->clear_child_tid = tidptr;

 return task_pid_vnr(current);
}

static void rt_mutex_init_task(struct task_struct *p)
{
 raw_spin_lock_init(&p->pi_lock);
 p->pi_waiters = RB_ROOT;
 p->pi_waiters_leftmost = NULL;
 p->pi_blocked_on = NULL;
}




static void posix_cpu_timers_init(struct task_struct *tsk)
{
 tsk->cputime_expires.prof_exp = 0;
 tsk->cputime_expires.virt_exp = 0;
 tsk->cputime_expires.sched_exp = 0;
 INIT_LIST_HEAD(&tsk->cpu_timers[0]);
 INIT_LIST_HEAD(&tsk->cpu_timers[1]);
 INIT_LIST_HEAD(&tsk->cpu_timers[2]);
}

static inline void
init_task_pid(struct task_struct *task, enum pid_type type, struct pid *pid)
{
  task->pids[type].pid = pid;
}
static struct task_struct *copy_process(unsigned long clone_flags,
     unsigned long stack_start,
     unsigned long stack_size,
     int __user *child_tidptr,
     struct pid *pid,
     int trace,
     unsigned long tls,
     int node)
{
 int retval;
 struct task_struct *p;

 if ((clone_flags & (CLONE_NEWNS|CLONE_FS)) == (CLONE_NEWNS|CLONE_FS))
  return ERR_PTR(-EINVAL);

 if ((clone_flags & (CLONE_NEWUSER|CLONE_FS)) == (CLONE_NEWUSER|CLONE_FS))
  return ERR_PTR(-EINVAL);





 if ((clone_flags & CLONE_THREAD) && !(clone_flags & CLONE_SIGHAND))
  return ERR_PTR(-EINVAL);






 if ((clone_flags & CLONE_SIGHAND) && !(clone_flags & CLONE_VM))
  return ERR_PTR(-EINVAL);







 if ((clone_flags & CLONE_PARENT) &&
    current->signal->flags & SIGNAL_UNKILLABLE)
  return ERR_PTR(-EINVAL);





 if (clone_flags & CLONE_THREAD) {
  if ((clone_flags & (CLONE_NEWUSER | CLONE_NEWPID)) ||
      (task_active_pid_ns(current) !=
    current->nsproxy->pid_ns_for_children))
   return ERR_PTR(-EINVAL);
 }

 retval = security_task_create(clone_flags);
 if (retval)
  goto fork_out;

 retval = -ENOMEM;
 p = dup_task_struct(current, node);
 if (!p)
  goto fork_out;

 ftrace_graph_init_task(p);

 rt_mutex_init_task(p);

 DEBUG_LOCKS_WARN_ON(!p->hardirqs_enabled);
 DEBUG_LOCKS_WARN_ON(!p->softirqs_enabled);
 retval = -EAGAIN;
 if (atomic_read(&p->real_cred->user->processes) >=
   task_rlimit(p, RLIMIT_NPROC)) {
  if (p->real_cred->user != INIT_USER &&
      !capable(CAP_SYS_RESOURCE) && !capable(CAP_SYS_ADMIN))
   goto bad_fork_free;
 }
 current->flags &= ~PF_NPROC_EXCEEDED;

 retval = copy_creds(p, clone_flags);
 if (retval < 0)
  goto bad_fork_free;






 retval = -EAGAIN;
 if (nr_threads >= max_threads)
  goto bad_fork_cleanup_count;

 delayacct_tsk_init(p);
 p->flags &= ~(PF_SUPERPRIV | PF_WQ_WORKER);
 p->flags |= PF_FORKNOEXEC;
 INIT_LIST_HEAD(&p->children);
 INIT_LIST_HEAD(&p->sibling);
 rcu_copy_process(p);
 p->vfork_done = NULL;
 spin_lock_init(&p->alloc_lock);

 init_sigpending(&p->pending);

 p->utime = p->stime = p->gtime = 0;
 p->utimescaled = p->stimescaled = 0;
 prev_cputime_init(&p->prev_cputime);

 seqcount_init(&p->vtime_seqcount);
 p->vtime_snap = 0;
 p->vtime_snap_whence = VTIME_INACTIVE;

 memset(&p->rss_stat, 0, sizeof(p->rss_stat));

 p->default_timer_slack_ns = current->timer_slack_ns;

 task_io_accounting_init(&p->ioac);
 acct_clear_integrals(p);

 posix_cpu_timers_init(p);

 p->start_time = ktime_get_ns();
 p->real_start_time = ktime_get_boot_ns();
 p->io_context = NULL;
 p->audit_context = NULL;
 threadgroup_change_begin(current);
 cgroup_fork(p);
 p->mempolicy = mpol_dup(p->mempolicy);
 if (IS_ERR(p->mempolicy)) {
  retval = PTR_ERR(p->mempolicy);
  p->mempolicy = NULL;
  goto bad_fork_cleanup_threadgroup_lock;
 }
 p->cpuset_mem_spread_rotor = NUMA_NO_NODE;
 p->cpuset_slab_spread_rotor = NUMA_NO_NODE;
 seqcount_init(&p->mems_allowed_seq);
 p->irq_events = 0;
 p->hardirqs_enabled = 0;
 p->hardirq_enable_ip = 0;
 p->hardirq_enable_event = 0;
 p->hardirq_disable_ip = _THIS_IP_;
 p->hardirq_disable_event = 0;
 p->softirqs_enabled = 1;
 p->softirq_enable_ip = _THIS_IP_;
 p->softirq_enable_event = 0;
 p->softirq_disable_ip = 0;
 p->softirq_disable_event = 0;
 p->hardirq_context = 0;
 p->softirq_context = 0;

 p->pagefault_disabled = 0;

 p->lockdep_depth = 0;
 p->curr_chain_key = 0;
 p->lockdep_recursion = 0;

 p->blocked_on = NULL;
 p->sequential_io = 0;
 p->sequential_io_avg = 0;


 retval = sched_fork(clone_flags, p);
 if (retval)
  goto bad_fork_cleanup_policy;

 retval = perf_event_init_task(p);
 if (retval)
  goto bad_fork_cleanup_policy;
 retval = audit_alloc(p);
 if (retval)
  goto bad_fork_cleanup_perf;

 shm_init_task(p);
 retval = copy_semundo(clone_flags, p);
 if (retval)
  goto bad_fork_cleanup_audit;
 retval = copy_files(clone_flags, p);
 if (retval)
  goto bad_fork_cleanup_semundo;
 retval = copy_fs(clone_flags, p);
 if (retval)
  goto bad_fork_cleanup_files;
 retval = copy_sighand(clone_flags, p);
 if (retval)
  goto bad_fork_cleanup_fs;
 retval = copy_signal(clone_flags, p);
 if (retval)
  goto bad_fork_cleanup_sighand;
 retval = copy_mm(clone_flags, p);
 if (retval)
  goto bad_fork_cleanup_signal;
 retval = copy_namespaces(clone_flags, p);
 if (retval)
  goto bad_fork_cleanup_mm;
 retval = copy_io(clone_flags, p);
 if (retval)
  goto bad_fork_cleanup_namespaces;
 retval = copy_thread_tls(clone_flags, stack_start, stack_size, p, tls);
 if (retval)
  goto bad_fork_cleanup_io;

 if (pid != &init_struct_pid) {
  pid = alloc_pid(p->nsproxy->pid_ns_for_children);
  if (IS_ERR(pid)) {
   retval = PTR_ERR(pid);
   goto bad_fork_cleanup_thread;
  }
 }

 p->set_child_tid = (clone_flags & CLONE_CHILD_SETTID) ? child_tidptr : NULL;



 p->clear_child_tid = (clone_flags & CLONE_CHILD_CLEARTID) ? child_tidptr : NULL;
 p->plug = NULL;
 p->robust_list = NULL;
 p->compat_robust_list = NULL;
 INIT_LIST_HEAD(&p->pi_state_list);
 p->pi_state_cache = NULL;



 if ((clone_flags & (CLONE_VM|CLONE_VFORK)) == CLONE_VM)
  sas_ss_reset(p);





 user_disable_single_step(p);
 clear_tsk_thread_flag(p, TIF_SYSCALL_TRACE);
 clear_tsk_thread_flag(p, TIF_SYSCALL_EMU);
 clear_all_latency_tracing(p);


 p->pid = pid_nr(pid);
 if (clone_flags & CLONE_THREAD) {
  p->exit_signal = -1;
  p->group_leader = current->group_leader;
  p->tgid = current->tgid;
 } else {
  if (clone_flags & CLONE_PARENT)
   p->exit_signal = current->group_leader->exit_signal;
  else
   p->exit_signal = (clone_flags & CSIGNAL);
  p->group_leader = p;
  p->tgid = p->pid;
 }

 p->nr_dirtied = 0;
 p->nr_dirtied_pause = 128 >> (PAGE_SHIFT - 10);
 p->dirty_paused_when = 0;

 p->pdeath_signal = 0;
 INIT_LIST_HEAD(&p->thread_group);
 p->task_works = NULL;







 retval = cgroup_can_fork(p);
 if (retval)
  goto bad_fork_free_pid;





 write_lock_irq(&tasklist_lock);


 if (clone_flags & (CLONE_PARENT|CLONE_THREAD)) {
  p->real_parent = current->real_parent;
  p->parent_exec_id = current->parent_exec_id;
 } else {
  p->real_parent = current;
  p->parent_exec_id = current->self_exec_id;
 }

 spin_lock(&current->sighand->siglock);





 copy_seccomp(p);
 recalc_sigpending();
 if (signal_pending(current)) {
  spin_unlock(&current->sighand->siglock);
  write_unlock_irq(&tasklist_lock);
  retval = -ERESTARTNOINTR;
  goto bad_fork_cancel_cgroup;
 }

 if (likely(p->pid)) {
  ptrace_init_task(p, (clone_flags & CLONE_PTRACE) || trace);

  init_task_pid(p, PIDTYPE_PID, pid);
  if (thread_group_leader(p)) {
   init_task_pid(p, PIDTYPE_PGID, task_pgrp(current));
   init_task_pid(p, PIDTYPE_SID, task_session(current));

   if (is_child_reaper(pid)) {
    ns_of_pid(pid)->child_reaper = p;
    p->signal->flags |= SIGNAL_UNKILLABLE;
   }

   p->signal->leader_pid = pid;
   p->signal->tty = tty_kref_get(current->signal->tty);
   list_add_tail(&p->sibling, &p->real_parent->children);
   list_add_tail_rcu(&p->tasks, &init_task.tasks);
   attach_pid(p, PIDTYPE_PGID);
   attach_pid(p, PIDTYPE_SID);
   __this_cpu_inc(process_counts);
  } else {
   current->signal->nr_threads++;
   atomic_inc(&current->signal->live);
   atomic_inc(&current->signal->sigcnt);
   list_add_tail_rcu(&p->thread_group,
       &p->group_leader->thread_group);
   list_add_tail_rcu(&p->thread_node,
       &p->signal->thread_head);
  }
  attach_pid(p, PIDTYPE_PID);
  nr_threads++;
 }

 total_forks++;
 spin_unlock(&current->sighand->siglock);
 syscall_tracepoint_update(p);
 write_unlock_irq(&tasklist_lock);

 proc_fork_connector(p);
 cgroup_post_fork(p);
 threadgroup_change_end(current);
 perf_event_fork(p);

 trace_task_newtask(p, clone_flags);
 uprobe_copy_process(p, clone_flags);

 return p;

bad_fork_cancel_cgroup:
 cgroup_cancel_fork(p);
bad_fork_free_pid:
 if (pid != &init_struct_pid)
  free_pid(pid);
bad_fork_cleanup_thread:
 exit_thread(p);
bad_fork_cleanup_io:
 if (p->io_context)
  exit_io_context(p);
bad_fork_cleanup_namespaces:
 exit_task_namespaces(p);
bad_fork_cleanup_mm:
 if (p->mm)
  mmput(p->mm);
bad_fork_cleanup_signal:
 if (!(clone_flags & CLONE_THREAD))
  free_signal_struct(p->signal);
bad_fork_cleanup_sighand:
 __cleanup_sighand(p->sighand);
bad_fork_cleanup_fs:
 exit_fs(p);
bad_fork_cleanup_files:
 exit_files(p);
bad_fork_cleanup_semundo:
 exit_sem(p);
bad_fork_cleanup_audit:
 audit_free(p);
bad_fork_cleanup_perf:
 perf_event_free_task(p);
bad_fork_cleanup_policy:
 mpol_put(p->mempolicy);
bad_fork_cleanup_threadgroup_lock:
 threadgroup_change_end(current);
 delayacct_tsk_free(p);
bad_fork_cleanup_count:
 atomic_dec(&p->cred->user->processes);
 exit_creds(p);
bad_fork_free:
 free_task(p);
fork_out:
 return ERR_PTR(retval);
}

static inline void init_idle_pids(struct pid_link *links)
{
 enum pid_type type;

 for (type = PIDTYPE_PID; type < PIDTYPE_MAX; ++type) {
  INIT_HLIST_NODE(&links[type].node);
  links[type].pid = &init_struct_pid;
 }
}

struct task_struct *fork_idle(int cpu)
{
 struct task_struct *task;
 task = copy_process(CLONE_VM, 0, 0, NULL, &init_struct_pid, 0, 0,
       cpu_to_node(cpu));
 if (!IS_ERR(task)) {
  init_idle_pids(task->pids);
  init_idle(task, cpu);
 }

 return task;
}







long _do_fork(unsigned long clone_flags,
       unsigned long stack_start,
       unsigned long stack_size,
       int __user *parent_tidptr,
       int __user *child_tidptr,
       unsigned long tls)
{
 struct task_struct *p;
 int trace = 0;
 long nr;







 if (!(clone_flags & CLONE_UNTRACED)) {
  if (clone_flags & CLONE_VFORK)
   trace = PTRACE_EVENT_VFORK;
  else if ((clone_flags & CSIGNAL) != SIGCHLD)
   trace = PTRACE_EVENT_CLONE;
  else
   trace = PTRACE_EVENT_FORK;

  if (likely(!ptrace_event_enabled(current, trace)))
   trace = 0;
 }

 p = copy_process(clone_flags, stack_start, stack_size,
    child_tidptr, NULL, trace, tls, NUMA_NO_NODE);




 if (!IS_ERR(p)) {
  struct completion vfork;
  struct pid *pid;

  trace_sched_process_fork(current, p);

  pid = get_task_pid(p, PIDTYPE_PID);
  nr = pid_vnr(pid);

  if (clone_flags & CLONE_PARENT_SETTID)
   put_user(nr, parent_tidptr);

  if (clone_flags & CLONE_VFORK) {
   p->vfork_done = &vfork;
   init_completion(&vfork);
   get_task_struct(p);
  }

  wake_up_new_task(p);


  if (unlikely(trace))
   ptrace_event_pid(trace, pid);

  if (clone_flags & CLONE_VFORK) {
   if (!wait_for_vfork_done(p, &vfork))
    ptrace_event_pid(PTRACE_EVENT_VFORK_DONE, pid);
  }

  put_pid(pid);
 } else {
  nr = PTR_ERR(p);
 }
 return nr;
}



long do_fork(unsigned long clone_flags,
       unsigned long stack_start,
       unsigned long stack_size,
       int __user *parent_tidptr,
       int __user *child_tidptr)
{
 return _do_fork(clone_flags, stack_start, stack_size,
   parent_tidptr, child_tidptr, 0);
}




pid_t kernel_thread(int (*fn)(void *), void *arg, unsigned long flags)
{
 return _do_fork(flags|CLONE_VM|CLONE_UNTRACED, (unsigned long)fn,
  (unsigned long)arg, NULL, NULL, 0);
}

SYSCALL_DEFINE0(fork)
{
 return _do_fork(SIGCHLD, 0, 0, NULL, NULL, 0);

 return -EINVAL;
}

SYSCALL_DEFINE0(vfork)
{
 return _do_fork(CLONE_VFORK | CLONE_VM | SIGCHLD, 0,
   0, NULL, NULL, 0);
}

SYSCALL_DEFINE5(clone, unsigned long, clone_flags, unsigned long, newsp,
   int __user *, parent_tidptr,
   unsigned long, tls,
   int __user *, child_tidptr)
SYSCALL_DEFINE5(clone, unsigned long, newsp, unsigned long, clone_flags,
   int __user *, parent_tidptr,
   int __user *, child_tidptr,
   unsigned long, tls)
SYSCALL_DEFINE6(clone, unsigned long, clone_flags, unsigned long, newsp,
  int, stack_size,
  int __user *, parent_tidptr,
  int __user *, child_tidptr,
  unsigned long, tls)
SYSCALL_DEFINE5(clone, unsigned long, clone_flags, unsigned long, newsp,
   int __user *, parent_tidptr,
   int __user *, child_tidptr,
   unsigned long, tls)
{
 return _do_fork(clone_flags, newsp, 0, parent_tidptr, child_tidptr, tls);
}


static void sighand_ctor(void *data)
{
 struct sighand_struct *sighand = data;

 spin_lock_init(&sighand->siglock);
 init_waitqueue_head(&sighand->signalfd_wqh);
}

void __init proc_caches_init(void)
{
 sighand_cachep = kmem_cache_create("sighand_cache",
   sizeof(struct sighand_struct), 0,
   SLAB_HWCACHE_ALIGN|SLAB_PANIC|SLAB_DESTROY_BY_RCU|
   SLAB_NOTRACK|SLAB_ACCOUNT, sighand_ctor);
 signal_cachep = kmem_cache_create("signal_cache",
   sizeof(struct signal_struct), 0,
   SLAB_HWCACHE_ALIGN|SLAB_PANIC|SLAB_NOTRACK|SLAB_ACCOUNT,
   NULL);
 files_cachep = kmem_cache_create("files_cache",
   sizeof(struct files_struct), 0,
   SLAB_HWCACHE_ALIGN|SLAB_PANIC|SLAB_NOTRACK|SLAB_ACCOUNT,
   NULL);
 fs_cachep = kmem_cache_create("fs_cache",
   sizeof(struct fs_struct), 0,
   SLAB_HWCACHE_ALIGN|SLAB_PANIC|SLAB_NOTRACK|SLAB_ACCOUNT,
   NULL);







 mm_cachep = kmem_cache_create("mm_struct",
   sizeof(struct mm_struct), ARCH_MIN_MMSTRUCT_ALIGN,
   SLAB_HWCACHE_ALIGN|SLAB_PANIC|SLAB_NOTRACK|SLAB_ACCOUNT,
   NULL);
 vm_area_cachep = KMEM_CACHE(vm_area_struct, SLAB_PANIC|SLAB_ACCOUNT);
 mmap_init();
 nsproxy_cache_init();
}




static int check_unshare_flags(unsigned long unshare_flags)
{
 if (unshare_flags & ~(CLONE_THREAD|CLONE_FS|CLONE_NEWNS|CLONE_SIGHAND|
    CLONE_VM|CLONE_FILES|CLONE_SYSVSEM|
    CLONE_NEWUTS|CLONE_NEWIPC|CLONE_NEWNET|
    CLONE_NEWUSER|CLONE_NEWPID|CLONE_NEWCGROUP))
  return -EINVAL;






 if (unshare_flags & (CLONE_THREAD | CLONE_SIGHAND | CLONE_VM)) {
  if (!thread_group_empty(current))
   return -EINVAL;
 }
 if (unshare_flags & (CLONE_SIGHAND | CLONE_VM)) {
  if (atomic_read(&current->sighand->count) > 1)
   return -EINVAL;
 }
 if (unshare_flags & CLONE_VM) {
  if (!current_is_single_threaded())
   return -EINVAL;
 }

 return 0;
}




static int unshare_fs(unsigned long unshare_flags, struct fs_struct **new_fsp)
{
 struct fs_struct *fs = current->fs;

 if (!(unshare_flags & CLONE_FS) || !fs)
  return 0;


 if (fs->users == 1)
  return 0;

 *new_fsp = copy_fs_struct(fs);
 if (!*new_fsp)
  return -ENOMEM;

 return 0;
}




static int unshare_fd(unsigned long unshare_flags, struct files_struct **new_fdp)
{
 struct files_struct *fd = current->files;
 int error = 0;

 if ((unshare_flags & CLONE_FILES) &&
     (fd && atomic_read(&fd->count) > 1)) {
  *new_fdp = dup_fd(fd, &error);
  if (!*new_fdp)
   return error;
 }

 return 0;
}
SYSCALL_DEFINE1(unshare, unsigned long, unshare_flags)
{
 struct fs_struct *fs, *new_fs = NULL;
 struct files_struct *fd, *new_fd = NULL;
 struct cred *new_cred = NULL;
 struct nsproxy *new_nsproxy = NULL;
 int do_sysvsem = 0;
 int err;





 if (unshare_flags & CLONE_NEWUSER)
  unshare_flags |= CLONE_THREAD | CLONE_FS;



 if (unshare_flags & CLONE_VM)
  unshare_flags |= CLONE_SIGHAND;



 if (unshare_flags & CLONE_SIGHAND)
  unshare_flags |= CLONE_THREAD;



 if (unshare_flags & CLONE_NEWNS)
  unshare_flags |= CLONE_FS;

 err = check_unshare_flags(unshare_flags);
 if (err)
  goto bad_unshare_out;





 if (unshare_flags & (CLONE_NEWIPC|CLONE_SYSVSEM))
  do_sysvsem = 1;
 err = unshare_fs(unshare_flags, &new_fs);
 if (err)
  goto bad_unshare_out;
 err = unshare_fd(unshare_flags, &new_fd);
 if (err)
  goto bad_unshare_cleanup_fs;
 err = unshare_userns(unshare_flags, &new_cred);
 if (err)
  goto bad_unshare_cleanup_fd;
 err = unshare_nsproxy_namespaces(unshare_flags, &new_nsproxy,
      new_cred, new_fs);
 if (err)
  goto bad_unshare_cleanup_cred;

 if (new_fs || new_fd || do_sysvsem || new_cred || new_nsproxy) {
  if (do_sysvsem) {



   exit_sem(current);
  }
  if (unshare_flags & CLONE_NEWIPC) {

   exit_shm(current);
   shm_init_task(current);
  }

  if (new_nsproxy)
   switch_task_namespaces(current, new_nsproxy);

  task_lock(current);

  if (new_fs) {
   fs = current->fs;
   spin_lock(&fs->lock);
   current->fs = new_fs;
   if (--fs->users)
    new_fs = NULL;
   else
    new_fs = fs;
   spin_unlock(&fs->lock);
  }

  if (new_fd) {
   fd = current->files;
   current->files = new_fd;
   new_fd = fd;
  }

  task_unlock(current);

  if (new_cred) {

   commit_creds(new_cred);
   new_cred = NULL;
  }
 }

bad_unshare_cleanup_cred:
 if (new_cred)
  put_cred(new_cred);
bad_unshare_cleanup_fd:
 if (new_fd)
  put_files_struct(new_fd);

bad_unshare_cleanup_fs:
 if (new_fs)
  free_fs_struct(new_fs);

bad_unshare_out:
 return err;
}







int unshare_files(struct files_struct **displaced)
{
 struct task_struct *task = current;
 struct files_struct *copy = NULL;
 int error;

 error = unshare_fd(CLONE_FILES, &copy);
 if (error || !copy) {
  *displaced = NULL;
  return error;
 }
 *displaced = task->files;
 task_lock(task);
 task->files = copy;
 task_unlock(task);
 return 0;
}

int sysctl_max_threads(struct ctl_table *table, int write,
         void __user *buffer, size_t *lenp, loff_t *ppos)
{
 struct ctl_table t;
 int ret;
 int threads = max_threads;
 int min = MIN_THREADS;
 int max = MAX_THREADS;

 t = *table;
 t.data = &threads;
 t.extra1 = &min;
 t.extra2 = &max;

 ret = proc_dointvec_minmax(&t, write, buffer, lenp, ppos);
 if (ret || !write)
  return ret;

 set_max_threads(threads);

 return 0;
}








atomic_t system_freezing_cnt = ATOMIC_INIT(0);
EXPORT_SYMBOL(system_freezing_cnt);


bool pm_freezing;
bool pm_nosig_freezing;





EXPORT_SYMBOL_GPL(pm_freezing);


static DEFINE_SPINLOCK(freezer_lock);
bool freezing_slow_path(struct task_struct *p)
{
 if (p->flags & (PF_NOFREEZE | PF_SUSPEND_TASK))
  return false;

 if (test_thread_flag(TIF_MEMDIE))
  return false;

 if (pm_nosig_freezing || cgroup_freezing(p))
  return true;

 if (pm_freezing && !(p->flags & PF_KTHREAD))
  return true;

 return false;
}
EXPORT_SYMBOL(freezing_slow_path);


bool __refrigerator(bool check_kthr_stop)
{


 bool was_frozen = false;
 long save = current->state;

 pr_debug("%s entered refrigerator\n", current->comm);

 for (;;) {
  set_current_state(TASK_UNINTERRUPTIBLE);

  spin_lock_irq(&freezer_lock);
  current->flags |= PF_FROZEN;
  if (!freezing(current) ||
      (check_kthr_stop && kthread_should_stop()))
   current->flags &= ~PF_FROZEN;
  spin_unlock_irq(&freezer_lock);

  if (!(current->flags & PF_FROZEN))
   break;
  was_frozen = true;
  schedule();
 }

 pr_debug("%s left refrigerator\n", current->comm);






 set_current_state(save);

 return was_frozen;
}
EXPORT_SYMBOL(__refrigerator);

static void fake_signal_wake_up(struct task_struct *p)
{
 unsigned long flags;

 if (lock_task_sighand(p, &flags)) {
  signal_wake_up(p, 0);
  unlock_task_sighand(p, &flags);
 }
}
bool freeze_task(struct task_struct *p)
{
 unsigned long flags;
 if (freezer_should_skip(p))
  return false;

 spin_lock_irqsave(&freezer_lock, flags);
 if (!freezing(p) || frozen(p)) {
  spin_unlock_irqrestore(&freezer_lock, flags);
  return false;
 }

 if (!(p->flags & PF_KTHREAD))
  fake_signal_wake_up(p);
 else
  wake_up_state(p, TASK_INTERRUPTIBLE);

 spin_unlock_irqrestore(&freezer_lock, flags);
 return true;
}

void __thaw_task(struct task_struct *p)
{
 unsigned long flags;

 spin_lock_irqsave(&freezer_lock, flags);
 if (frozen(p))
  wake_up_process(p);
 spin_unlock_irqrestore(&freezer_lock, flags);
}






bool set_freezable(void)
{
 might_sleep();






 spin_lock_irq(&freezer_lock);
 current->flags &= ~PF_NOFREEZE;
 spin_unlock_irq(&freezer_lock);

 return try_to_freeze();
}
EXPORT_SYMBOL(set_freezable);


int __read_mostly futex_cmpxchg_enabled;









struct futex_pi_state {




 struct list_head list;




 struct rt_mutex pi_mutex;

 struct task_struct *owner;
 atomic_t refcount;

 union futex_key key;
};
struct futex_q {
 struct plist_node list;

 struct task_struct *task;
 spinlock_t *lock_ptr;
 union futex_key key;
 struct futex_pi_state *pi_state;
 struct rt_mutex_waiter *rt_waiter;
 union futex_key *requeue_pi_key;
 u32 bitset;
};

static const struct futex_q futex_q_init = {

 .key = FUTEX_KEY_INIT,
 .bitset = FUTEX_BITSET_MATCH_ANY
};






struct futex_hash_bucket {
 atomic_t waiters;
 spinlock_t lock;
 struct plist_head chain;
} ____cacheline_aligned_in_smp;






static struct {
 struct futex_hash_bucket *queues;
 unsigned long hashsize;
} __futex_data __read_mostly __aligned(2*sizeof(long));






static struct {
 struct fault_attr attr;

 bool ignore_private;
} fail_futex = {
 .attr = FAULT_ATTR_INITIALIZER,
 .ignore_private = false,
};

static int __init setup_fail_futex(char *str)
{
 return setup_fault_attr(&fail_futex.attr, str);
}
__setup("fail_futex=", setup_fail_futex);

static bool should_fail_futex(bool fshared)
{
 if (fail_futex.ignore_private && !fshared)
  return false;

 return should_fail(&fail_futex.attr, 1);
}


static int __init fail_futex_debugfs(void)
{
 umode_t mode = S_IFREG | S_IRUSR | S_IWUSR;
 struct dentry *dir;

 dir = fault_create_debugfs_attr("fail_futex", NULL,
     &fail_futex.attr);
 if (IS_ERR(dir))
  return PTR_ERR(dir);

 if (!debugfs_create_bool("ignore-private", mode, dir,
     &fail_futex.ignore_private)) {
  debugfs_remove_recursive(dir);
  return -ENOMEM;
 }

 return 0;
}

late_initcall(fail_futex_debugfs);


static inline bool should_fail_futex(bool fshared)
{
 return false;
}

static inline void futex_get_mm(union futex_key *key)
{
 atomic_inc(&key->private.mm->mm_count);





 smp_mb__after_atomic();
}




static inline void hb_waiters_inc(struct futex_hash_bucket *hb)
{
 atomic_inc(&hb->waiters);



 smp_mb__after_atomic();
}





static inline void hb_waiters_dec(struct futex_hash_bucket *hb)
{
 atomic_dec(&hb->waiters);
}

static inline int hb_waiters_pending(struct futex_hash_bucket *hb)
{
 return atomic_read(&hb->waiters);
 return 1;
}




static struct futex_hash_bucket *hash_futex(union futex_key *key)
{
 u32 hash = jhash2((u32*)&key->both.word,
     (sizeof(key->both.word)+sizeof(key->both.ptr))/4,
     key->both.offset);
 return &futex_queues[hash & (futex_hashsize - 1)];
}




static inline int match_futex(union futex_key *key1, union futex_key *key2)
{
 return (key1 && key2
  && key1->both.word == key2->both.word
  && key1->both.ptr == key2->both.ptr
  && key1->both.offset == key2->both.offset);
}






static void get_futex_key_refs(union futex_key *key)
{
 if (!key->both.ptr)
  return;

 switch (key->both.offset & (FUT_OFF_INODE|FUT_OFF_MMSHARED)) {
 case FUT_OFF_INODE:
  ihold(key->shared.inode);
  break;
 case FUT_OFF_MMSHARED:
  futex_get_mm(key);
  break;
 default:





  smp_mb();
 }
}







static void drop_futex_key_refs(union futex_key *key)
{
 if (!key->both.ptr) {

  WARN_ON_ONCE(1);
  return;
 }

 switch (key->both.offset & (FUT_OFF_INODE|FUT_OFF_MMSHARED)) {
 case FUT_OFF_INODE:
  iput(key->shared.inode);
  break;
 case FUT_OFF_MMSHARED:
  mmdrop(key->private.mm);
  break;
 }
}
static int
get_futex_key(u32 __user *uaddr, int fshared, union futex_key *key, int rw)
{
 unsigned long address = (unsigned long)uaddr;
 struct mm_struct *mm = current->mm;
 struct page *page, *tail;
 struct address_space *mapping;
 int err, ro = 0;




 key->both.offset = address % PAGE_SIZE;
 if (unlikely((address % sizeof(u32)) != 0))
  return -EINVAL;
 address -= key->both.offset;

 if (unlikely(!access_ok(rw, uaddr, sizeof(u32))))
  return -EFAULT;

 if (unlikely(should_fail_futex(fshared)))
  return -EFAULT;
 if (!fshared) {
  key->private.mm = mm;
  key->private.address = address;
  get_futex_key_refs(key);
  return 0;
 }

again:

 if (unlikely(should_fail_futex(fshared)))
  return -EFAULT;

 err = get_user_pages_fast(address, 1, 1, &page);




 if (err == -EFAULT && rw == VERIFY_READ) {
  err = get_user_pages_fast(address, 1, 0, &page);
  ro = 1;
 }
 if (err < 0)
  return err;
 else
  err = 0;
 tail = page;
 page = compound_head(page);
 mapping = READ_ONCE(page->mapping);
 if (unlikely(!mapping)) {
  int shmem_swizzled;






  lock_page(page);
  shmem_swizzled = PageSwapCache(page) || page->mapping;
  unlock_page(page);
  put_page(page);

  if (shmem_swizzled)
   goto again;

  return -EFAULT;
 }
 if (PageAnon(page)) {




  if (unlikely(should_fail_futex(fshared)) || ro) {
   err = -EFAULT;
   goto out;
  }

  key->both.offset |= FUT_OFF_MMSHARED;
  key->private.mm = mm;
  key->private.address = address;

  get_futex_key_refs(key);

 } else {
  struct inode *inode;
  rcu_read_lock();

  if (READ_ONCE(page->mapping) != mapping) {
   rcu_read_unlock();
   put_page(page);

   goto again;
  }

  inode = READ_ONCE(mapping->host);
  if (!inode) {
   rcu_read_unlock();
   put_page(page);

   goto again;
  }
  if (WARN_ON_ONCE(!atomic_inc_not_zero(&inode->i_count))) {
   rcu_read_unlock();
   put_page(page);

   goto again;
  }


  if (WARN_ON_ONCE(inode->i_mapping != mapping)) {
   err = -EFAULT;
   rcu_read_unlock();
   iput(inode);

   goto out;
  }

  key->both.offset |= FUT_OFF_INODE;
  key->shared.inode = inode;
  key->shared.pgoff = basepage_index(tail);
  rcu_read_unlock();
 }

out:
 put_page(page);
 return err;
}

static inline void put_futex_key(union futex_key *key)
{
 drop_futex_key_refs(key);
}
static int fault_in_user_writeable(u32 __user *uaddr)
{
 struct mm_struct *mm = current->mm;
 int ret;

 down_read(&mm->mmap_sem);
 ret = fixup_user_fault(current, mm, (unsigned long)uaddr,
          FAULT_FLAG_WRITE, NULL);
 up_read(&mm->mmap_sem);

 return ret < 0 ? ret : 0;
}
static struct futex_q *futex_top_waiter(struct futex_hash_bucket *hb,
     union futex_key *key)
{
 struct futex_q *this;

 plist_for_each_entry(this, &hb->chain, list) {
  if (match_futex(&this->key, key))
   return this;
 }
 return NULL;
}

static int cmpxchg_futex_value_locked(u32 *curval, u32 __user *uaddr,
          u32 uval, u32 newval)
{
 int ret;

 pagefault_disable();
 ret = futex_atomic_cmpxchg_inatomic(curval, uaddr, uval, newval);
 pagefault_enable();

 return ret;
}

static int get_futex_value_locked(u32 *dest, u32 __user *from)
{
 int ret;

 pagefault_disable();
 ret = __get_user(*dest, from);
 pagefault_enable();

 return ret ? -EFAULT : 0;
}





static int refill_pi_state_cache(void)
{
 struct futex_pi_state *pi_state;

 if (likely(current->pi_state_cache))
  return 0;

 pi_state = kzalloc(sizeof(*pi_state), GFP_KERNEL);

 if (!pi_state)
  return -ENOMEM;

 INIT_LIST_HEAD(&pi_state->list);

 pi_state->owner = NULL;
 atomic_set(&pi_state->refcount, 1);
 pi_state->key = FUTEX_KEY_INIT;

 current->pi_state_cache = pi_state;

 return 0;
}

static struct futex_pi_state * alloc_pi_state(void)
{
 struct futex_pi_state *pi_state = current->pi_state_cache;

 WARN_ON(!pi_state);
 current->pi_state_cache = NULL;

 return pi_state;
}







static void put_pi_state(struct futex_pi_state *pi_state)
{
 if (!pi_state)
  return;

 if (!atomic_dec_and_test(&pi_state->refcount))
  return;





 if (pi_state->owner) {
  raw_spin_lock_irq(&pi_state->owner->pi_lock);
  list_del_init(&pi_state->list);
  raw_spin_unlock_irq(&pi_state->owner->pi_lock);

  rt_mutex_proxy_unlock(&pi_state->pi_mutex, pi_state->owner);
 }

 if (current->pi_state_cache)
  kfree(pi_state);
 else {





  pi_state->owner = NULL;
  atomic_set(&pi_state->refcount, 1);
  current->pi_state_cache = pi_state;
 }
}





static struct task_struct * futex_find_get_task(pid_t pid)
{
 struct task_struct *p;

 rcu_read_lock();
 p = find_task_by_vpid(pid);
 if (p)
  get_task_struct(p);

 rcu_read_unlock();

 return p;
}






void exit_pi_state_list(struct task_struct *curr)
{
 struct list_head *next, *head = &curr->pi_state_list;
 struct futex_pi_state *pi_state;
 struct futex_hash_bucket *hb;
 union futex_key key = FUTEX_KEY_INIT;

 if (!futex_cmpxchg_enabled)
  return;





 raw_spin_lock_irq(&curr->pi_lock);
 while (!list_empty(head)) {

  next = head->next;
  pi_state = list_entry(next, struct futex_pi_state, list);
  key = pi_state->key;
  hb = hash_futex(&key);
  raw_spin_unlock_irq(&curr->pi_lock);

  spin_lock(&hb->lock);

  raw_spin_lock_irq(&curr->pi_lock);




  if (head->next != next) {
   spin_unlock(&hb->lock);
   continue;
  }

  WARN_ON(pi_state->owner != curr);
  WARN_ON(list_empty(&pi_state->list));
  list_del_init(&pi_state->list);
  pi_state->owner = NULL;
  raw_spin_unlock_irq(&curr->pi_lock);

  rt_mutex_unlock(&pi_state->pi_mutex);

  spin_unlock(&hb->lock);

  raw_spin_lock_irq(&curr->pi_lock);
 }
 raw_spin_unlock_irq(&curr->pi_lock);
}
static int attach_to_pi_state(u32 uval, struct futex_pi_state *pi_state,
         struct futex_pi_state **ps)
{
 pid_t pid = uval & FUTEX_TID_MASK;




 if (unlikely(!pi_state))
  return -EINVAL;

 WARN_ON(!atomic_read(&pi_state->refcount));




 if (uval & FUTEX_OWNER_DIED) {





  if (!pi_state->owner) {




   if (pid)
    return -EINVAL;



   goto out_state;
  }
  if (!pid)
   goto out_state;
 } else {




  if (!pi_state->owner)
   return -EINVAL;
 }






 if (pid != task_pid_vnr(pi_state->owner))
  return -EINVAL;
out_state:
 atomic_inc(&pi_state->refcount);
 *ps = pi_state;
 return 0;
}





static int attach_to_pi_owner(u32 uval, union futex_key *key,
         struct futex_pi_state **ps)
{
 pid_t pid = uval & FUTEX_TID_MASK;
 struct futex_pi_state *pi_state;
 struct task_struct *p;





 if (!pid)
  return -ESRCH;
 p = futex_find_get_task(pid);
 if (!p)
  return -ESRCH;

 if (unlikely(p->flags & PF_KTHREAD)) {
  put_task_struct(p);
  return -EPERM;
 }







 raw_spin_lock_irq(&p->pi_lock);
 if (unlikely(p->flags & PF_EXITING)) {





  int ret = (p->flags & PF_EXITPIDONE) ? -ESRCH : -EAGAIN;

  raw_spin_unlock_irq(&p->pi_lock);
  put_task_struct(p);
  return ret;
 }




 pi_state = alloc_pi_state();





 rt_mutex_init_proxy_locked(&pi_state->pi_mutex, p);


 pi_state->key = *key;

 WARN_ON(!list_empty(&pi_state->list));
 list_add(&pi_state->list, &p->pi_state_list);
 pi_state->owner = p;
 raw_spin_unlock_irq(&p->pi_lock);

 put_task_struct(p);

 *ps = pi_state;

 return 0;
}

static int lookup_pi_state(u32 uval, struct futex_hash_bucket *hb,
      union futex_key *key, struct futex_pi_state **ps)
{
 struct futex_q *match = futex_top_waiter(hb, key);





 if (match)
  return attach_to_pi_state(uval, match->pi_state, ps);





 return attach_to_pi_owner(uval, key, ps);
}

static int lock_pi_update_atomic(u32 __user *uaddr, u32 uval, u32 newval)
{
 u32 uninitialized_var(curval);

 if (unlikely(should_fail_futex(true)))
  return -EFAULT;

 if (unlikely(cmpxchg_futex_value_locked(&curval, uaddr, uval, newval)))
  return -EFAULT;


 return curval != uval ? -EAGAIN : 0;
}
static int futex_lock_pi_atomic(u32 __user *uaddr, struct futex_hash_bucket *hb,
    union futex_key *key,
    struct futex_pi_state **ps,
    struct task_struct *task, int set_waiters)
{
 u32 uval, newval, vpid = task_pid_vnr(task);
 struct futex_q *match;
 int ret;





 if (get_futex_value_locked(&uval, uaddr))
  return -EFAULT;

 if (unlikely(should_fail_futex(true)))
  return -EFAULT;




 if ((unlikely((uval & FUTEX_TID_MASK) == vpid)))
  return -EDEADLK;

 if ((unlikely(should_fail_futex(true))))
  return -EDEADLK;





 match = futex_top_waiter(hb, key);
 if (match)
  return attach_to_pi_state(uval, match->pi_state, ps);







 if (!(uval & FUTEX_TID_MASK)) {




  newval = uval & FUTEX_OWNER_DIED;
  newval |= vpid;


  if (set_waiters)
   newval |= FUTEX_WAITERS;

  ret = lock_pi_update_atomic(uaddr, uval, newval);

  return ret < 0 ? ret : 1;
 }






 newval = uval | FUTEX_WAITERS;
 ret = lock_pi_update_atomic(uaddr, uval, newval);
 if (ret)
  return ret;





 return attach_to_pi_owner(uval, key, ps);
}







static void __unqueue_futex(struct futex_q *q)
{
 struct futex_hash_bucket *hb;

 if (WARN_ON_SMP(!q->lock_ptr || !spin_is_locked(q->lock_ptr))
     || WARN_ON(plist_node_empty(&q->list)))
  return;

 hb = container_of(q->lock_ptr, struct futex_hash_bucket, lock);
 plist_del(&q->list, &hb->chain);
 hb_waiters_dec(hb);
}







static void mark_wake_futex(struct wake_q_head *wake_q, struct futex_q *q)
{
 struct task_struct *p = q->task;

 if (WARN(q->pi_state || q->rt_waiter, "refusing to wake PI futex\n"))
  return;





 wake_q_add(wake_q, p);
 __unqueue_futex(q);






 smp_wmb();
 q->lock_ptr = NULL;
}

static int wake_futex_pi(u32 __user *uaddr, u32 uval, struct futex_q *this,
    struct futex_hash_bucket *hb)
{
 struct task_struct *new_owner;
 struct futex_pi_state *pi_state = this->pi_state;
 u32 uninitialized_var(curval), newval;
 WAKE_Q(wake_q);
 bool deboost;
 int ret = 0;

 if (!pi_state)
  return -EINVAL;





 if (pi_state->owner != current)
  return -EINVAL;

 raw_spin_lock_irq(&pi_state->pi_mutex.wait_lock);
 new_owner = rt_mutex_next_owner(&pi_state->pi_mutex);






 if (!new_owner)
  new_owner = this->task;






 newval = FUTEX_WAITERS | task_pid_vnr(new_owner);

 if (unlikely(should_fail_futex(true)))
  ret = -EFAULT;

 if (cmpxchg_futex_value_locked(&curval, uaddr, uval, newval)) {
  ret = -EFAULT;
 } else if (curval != uval) {






  if ((FUTEX_TID_MASK & curval) == uval)
   ret = -EAGAIN;
  else
   ret = -EINVAL;
 }
 if (ret) {
  raw_spin_unlock_irq(&pi_state->pi_mutex.wait_lock);
  return ret;
 }

 raw_spin_lock(&pi_state->owner->pi_lock);
 WARN_ON(list_empty(&pi_state->list));
 list_del_init(&pi_state->list);
 raw_spin_unlock(&pi_state->owner->pi_lock);

 raw_spin_lock(&new_owner->pi_lock);
 WARN_ON(!list_empty(&pi_state->list));
 list_add(&pi_state->list, &new_owner->pi_state_list);
 pi_state->owner = new_owner;
 raw_spin_unlock(&new_owner->pi_lock);

 raw_spin_unlock_irq(&pi_state->pi_mutex.wait_lock);

 deboost = rt_mutex_futex_unlock(&pi_state->pi_mutex, &wake_q);







 spin_unlock(&hb->lock);
 wake_up_q(&wake_q);
 if (deboost)
  rt_mutex_adjust_prio(current);

 return 0;
}




static inline void
double_lock_hb(struct futex_hash_bucket *hb1, struct futex_hash_bucket *hb2)
{
 if (hb1 <= hb2) {
  spin_lock(&hb1->lock);
  if (hb1 < hb2)
   spin_lock_nested(&hb2->lock, SINGLE_DEPTH_NESTING);
 } else {
  spin_lock(&hb2->lock);
  spin_lock_nested(&hb1->lock, SINGLE_DEPTH_NESTING);
 }
}

static inline void
double_unlock_hb(struct futex_hash_bucket *hb1, struct futex_hash_bucket *hb2)
{
 spin_unlock(&hb1->lock);
 if (hb1 != hb2)
  spin_unlock(&hb2->lock);
}




static int
futex_wake(u32 __user *uaddr, unsigned int flags, int nr_wake, u32 bitset)
{
 struct futex_hash_bucket *hb;
 struct futex_q *this, *next;
 union futex_key key = FUTEX_KEY_INIT;
 int ret;
 WAKE_Q(wake_q);

 if (!bitset)
  return -EINVAL;

 ret = get_futex_key(uaddr, flags & FLAGS_SHARED, &key, VERIFY_READ);
 if (unlikely(ret != 0))
  goto out;

 hb = hash_futex(&key);


 if (!hb_waiters_pending(hb))
  goto out_put_key;

 spin_lock(&hb->lock);

 plist_for_each_entry_safe(this, next, &hb->chain, list) {
  if (match_futex (&this->key, &key)) {
   if (this->pi_state || this->rt_waiter) {
    ret = -EINVAL;
    break;
   }


   if (!(this->bitset & bitset))
    continue;

   mark_wake_futex(&wake_q, this);
   if (++ret >= nr_wake)
    break;
  }
 }

 spin_unlock(&hb->lock);
 wake_up_q(&wake_q);
out_put_key:
 put_futex_key(&key);
out:
 return ret;
}





static int
futex_wake_op(u32 __user *uaddr1, unsigned int flags, u32 __user *uaddr2,
       int nr_wake, int nr_wake2, int op)
{
 union futex_key key1 = FUTEX_KEY_INIT, key2 = FUTEX_KEY_INIT;
 struct futex_hash_bucket *hb1, *hb2;
 struct futex_q *this, *next;
 int ret, op_ret;
 WAKE_Q(wake_q);

retry:
 ret = get_futex_key(uaddr1, flags & FLAGS_SHARED, &key1, VERIFY_READ);
 if (unlikely(ret != 0))
  goto out;
 ret = get_futex_key(uaddr2, flags & FLAGS_SHARED, &key2, VERIFY_WRITE);
 if (unlikely(ret != 0))
  goto out_put_key1;

 hb1 = hash_futex(&key1);
 hb2 = hash_futex(&key2);

retry_private:
 double_lock_hb(hb1, hb2);
 op_ret = futex_atomic_op_inuser(op, uaddr2);
 if (unlikely(op_ret < 0)) {

  double_unlock_hb(hb1, hb2);





  ret = op_ret;
  goto out_put_keys;

  if (unlikely(op_ret != -EFAULT)) {
   ret = op_ret;
   goto out_put_keys;
  }

  ret = fault_in_user_writeable(uaddr2);
  if (ret)
   goto out_put_keys;

  if (!(flags & FLAGS_SHARED))
   goto retry_private;

  put_futex_key(&key2);
  put_futex_key(&key1);
  goto retry;
 }

 plist_for_each_entry_safe(this, next, &hb1->chain, list) {
  if (match_futex (&this->key, &key1)) {
   if (this->pi_state || this->rt_waiter) {
    ret = -EINVAL;
    goto out_unlock;
   }
   mark_wake_futex(&wake_q, this);
   if (++ret >= nr_wake)
    break;
  }
 }

 if (op_ret > 0) {
  op_ret = 0;
  plist_for_each_entry_safe(this, next, &hb2->chain, list) {
   if (match_futex (&this->key, &key2)) {
    if (this->pi_state || this->rt_waiter) {
     ret = -EINVAL;
     goto out_unlock;
    }
    mark_wake_futex(&wake_q, this);
    if (++op_ret >= nr_wake2)
     break;
   }
  }
  ret += op_ret;
 }

out_unlock:
 double_unlock_hb(hb1, hb2);
 wake_up_q(&wake_q);
out_put_keys:
 put_futex_key(&key2);
out_put_key1:
 put_futex_key(&key1);
out:
 return ret;
}
static inline
void requeue_futex(struct futex_q *q, struct futex_hash_bucket *hb1,
     struct futex_hash_bucket *hb2, union futex_key *key2)
{





 if (likely(&hb1->chain != &hb2->chain)) {
  plist_del(&q->list, &hb1->chain);
  hb_waiters_dec(hb1);
  hb_waiters_inc(hb2);
  plist_add(&q->list, &hb2->chain);
  q->lock_ptr = &hb2->lock;
 }
 get_futex_key_refs(key2);
 q->key = *key2;
}
static inline
void requeue_pi_wake_futex(struct futex_q *q, union futex_key *key,
      struct futex_hash_bucket *hb)
{
 get_futex_key_refs(key);
 q->key = *key;

 __unqueue_futex(q);

 WARN_ON(!q->rt_waiter);
 q->rt_waiter = NULL;

 q->lock_ptr = &hb->lock;

 wake_up_state(q->task, TASK_NORMAL);
}
static int futex_proxy_trylock_atomic(u32 __user *pifutex,
     struct futex_hash_bucket *hb1,
     struct futex_hash_bucket *hb2,
     union futex_key *key1, union futex_key *key2,
     struct futex_pi_state **ps, int set_waiters)
{
 struct futex_q *top_waiter = NULL;
 u32 curval;
 int ret, vpid;

 if (get_futex_value_locked(&curval, pifutex))
  return -EFAULT;

 if (unlikely(should_fail_futex(true)))
  return -EFAULT;
 top_waiter = futex_top_waiter(hb1, key1);


 if (!top_waiter)
  return 0;


 if (!match_futex(top_waiter->requeue_pi_key, key2))
  return -EINVAL;






 vpid = task_pid_vnr(top_waiter->task);
 ret = futex_lock_pi_atomic(pifutex, hb2, key2, ps, top_waiter->task,
       set_waiters);
 if (ret == 1) {
  requeue_pi_wake_futex(top_waiter, key2, hb2);
  return vpid;
 }
 return ret;
}
static int futex_requeue(u32 __user *uaddr1, unsigned int flags,
    u32 __user *uaddr2, int nr_wake, int nr_requeue,
    u32 *cmpval, int requeue_pi)
{
 union futex_key key1 = FUTEX_KEY_INIT, key2 = FUTEX_KEY_INIT;
 int drop_count = 0, task_count = 0, ret;
 struct futex_pi_state *pi_state = NULL;
 struct futex_hash_bucket *hb1, *hb2;
 struct futex_q *this, *next;
 WAKE_Q(wake_q);

 if (requeue_pi) {




  if (uaddr1 == uaddr2)
   return -EINVAL;





  if (refill_pi_state_cache())
   return -ENOMEM;
  if (nr_wake != 1)
   return -EINVAL;
 }

retry:
 ret = get_futex_key(uaddr1, flags & FLAGS_SHARED, &key1, VERIFY_READ);
 if (unlikely(ret != 0))
  goto out;
 ret = get_futex_key(uaddr2, flags & FLAGS_SHARED, &key2,
       requeue_pi ? VERIFY_WRITE : VERIFY_READ);
 if (unlikely(ret != 0))
  goto out_put_key1;





 if (requeue_pi && match_futex(&key1, &key2)) {
  ret = -EINVAL;
  goto out_put_keys;
 }

 hb1 = hash_futex(&key1);
 hb2 = hash_futex(&key2);

retry_private:
 hb_waiters_inc(hb2);
 double_lock_hb(hb1, hb2);

 if (likely(cmpval != NULL)) {
  u32 curval;

  ret = get_futex_value_locked(&curval, uaddr1);

  if (unlikely(ret)) {
   double_unlock_hb(hb1, hb2);
   hb_waiters_dec(hb2);

   ret = get_user(curval, uaddr1);
   if (ret)
    goto out_put_keys;

   if (!(flags & FLAGS_SHARED))
    goto retry_private;

   put_futex_key(&key2);
   put_futex_key(&key1);
   goto retry;
  }
  if (curval != *cmpval) {
   ret = -EAGAIN;
   goto out_unlock;
  }
 }

 if (requeue_pi && (task_count - nr_wake < nr_requeue)) {






  ret = futex_proxy_trylock_atomic(uaddr2, hb1, hb2, &key1,
       &key2, &pi_state, nr_requeue);
  if (ret > 0) {
   WARN_ON(pi_state);
   drop_count++;
   task_count++;
   ret = lookup_pi_state(ret, hb2, &key2, &pi_state);
  }

  switch (ret) {
  case 0:

   break;


  case -EFAULT:
   double_unlock_hb(hb1, hb2);
   hb_waiters_dec(hb2);
   put_futex_key(&key2);
   put_futex_key(&key1);
   ret = fault_in_user_writeable(uaddr2);
   if (!ret)
    goto retry;
   goto out;
  case -EAGAIN:






   double_unlock_hb(hb1, hb2);
   hb_waiters_dec(hb2);
   put_futex_key(&key2);
   put_futex_key(&key1);
   cond_resched();
   goto retry;
  default:
   goto out_unlock;
  }
 }

 plist_for_each_entry_safe(this, next, &hb1->chain, list) {
  if (task_count - nr_wake >= nr_requeue)
   break;

  if (!match_futex(&this->key, &key1))
   continue;
  if ((requeue_pi && !this->rt_waiter) ||
      (!requeue_pi && this->rt_waiter) ||
      this->pi_state) {
   ret = -EINVAL;
   break;
  }






  if (++task_count <= nr_wake && !requeue_pi) {
   mark_wake_futex(&wake_q, this);
   continue;
  }


  if (requeue_pi && !match_futex(this->requeue_pi_key, &key2)) {
   ret = -EINVAL;
   break;
  }





  if (requeue_pi) {





   atomic_inc(&pi_state->refcount);
   this->pi_state = pi_state;
   ret = rt_mutex_start_proxy_lock(&pi_state->pi_mutex,
       this->rt_waiter,
       this->task);
   if (ret == 1) {
    requeue_pi_wake_futex(this, &key2, hb2);
    drop_count++;
    continue;
   } else if (ret) {
    this->pi_state = NULL;
    put_pi_state(pi_state);




    break;
   }
  }
  requeue_futex(this, hb1, hb2, &key2);
  drop_count++;
 }






 put_pi_state(pi_state);

out_unlock:
 double_unlock_hb(hb1, hb2);
 wake_up_q(&wake_q);
 hb_waiters_dec(hb2);







 while (--drop_count >= 0)
  drop_futex_key_refs(&key1);

out_put_keys:
 put_futex_key(&key2);
out_put_key1:
 put_futex_key(&key1);
out:
 return ret ? ret : task_count;
}


static inline struct futex_hash_bucket *queue_lock(struct futex_q *q)
 __acquires(&hb->lock)
{
 struct futex_hash_bucket *hb;

 hb = hash_futex(&q->key);
 hb_waiters_inc(hb);

 q->lock_ptr = &hb->lock;

 spin_lock(&hb->lock);
 return hb;
}

static inline void
queue_unlock(struct futex_hash_bucket *hb)
 __releases(&hb->lock)
{
 spin_unlock(&hb->lock);
 hb_waiters_dec(hb);
}
static inline void queue_me(struct futex_q *q, struct futex_hash_bucket *hb)
 __releases(&hb->lock)
{
 int prio;
 prio = min(current->normal_prio, MAX_RT_PRIO);

 plist_node_init(&q->list, prio);
 plist_add(&q->list, &hb->chain);
 q->task = current;
 spin_unlock(&hb->lock);
}
static int unqueue_me(struct futex_q *q)
{
 spinlock_t *lock_ptr;
 int ret = 0;


retry:





 lock_ptr = READ_ONCE(q->lock_ptr);
 if (lock_ptr != NULL) {
  spin_lock(lock_ptr);
  if (unlikely(lock_ptr != q->lock_ptr)) {
   spin_unlock(lock_ptr);
   goto retry;
  }
  __unqueue_futex(q);

  BUG_ON(q->pi_state);

  spin_unlock(lock_ptr);
  ret = 1;
 }

 drop_futex_key_refs(&q->key);
 return ret;
}






static void unqueue_me_pi(struct futex_q *q)
 __releases(q->lock_ptr)
{
 __unqueue_futex(q);

 BUG_ON(!q->pi_state);
 put_pi_state(q->pi_state);
 q->pi_state = NULL;

 spin_unlock(q->lock_ptr);
}







static int fixup_pi_state_owner(u32 __user *uaddr, struct futex_q *q,
    struct task_struct *newowner)
{
 u32 newtid = task_pid_vnr(newowner) | FUTEX_WAITERS;
 struct futex_pi_state *pi_state = q->pi_state;
 struct task_struct *oldowner = pi_state->owner;
 u32 uval, uninitialized_var(curval), newval;
 int ret;


 if (!pi_state->owner)
  newtid |= FUTEX_OWNER_DIED;
retry:
 if (get_futex_value_locked(&uval, uaddr))
  goto handle_fault;

 while (1) {
  newval = (uval & FUTEX_OWNER_DIED) | newtid;

  if (cmpxchg_futex_value_locked(&curval, uaddr, uval, newval))
   goto handle_fault;
  if (curval == uval)
   break;
  uval = curval;
 }





 if (pi_state->owner != NULL) {
  raw_spin_lock_irq(&pi_state->owner->pi_lock);
  WARN_ON(list_empty(&pi_state->list));
  list_del_init(&pi_state->list);
  raw_spin_unlock_irq(&pi_state->owner->pi_lock);
 }

 pi_state->owner = newowner;

 raw_spin_lock_irq(&newowner->pi_lock);
 WARN_ON(!list_empty(&pi_state->list));
 list_add(&pi_state->list, &newowner->pi_state_list);
 raw_spin_unlock_irq(&newowner->pi_lock);
 return 0;
handle_fault:
 spin_unlock(q->lock_ptr);

 ret = fault_in_user_writeable(uaddr);

 spin_lock(q->lock_ptr);




 if (pi_state->owner != oldowner)
  return 0;

 if (ret)
  return ret;

 goto retry;
}

static long futex_wait_restart(struct restart_block *restart);
static int fixup_owner(u32 __user *uaddr, struct futex_q *q, int locked)
{
 struct task_struct *owner;
 int ret = 0;

 if (locked) {




  if (q->pi_state->owner != current)
   ret = fixup_pi_state_owner(uaddr, q, current);
  goto out;
 }





 if (q->pi_state->owner == current) {





  if (rt_mutex_trylock(&q->pi_state->pi_mutex)) {
   locked = 1;
   goto out;
  }






  raw_spin_lock_irq(&q->pi_state->pi_mutex.wait_lock);
  owner = rt_mutex_owner(&q->pi_state->pi_mutex);
  if (!owner)
   owner = rt_mutex_next_owner(&q->pi_state->pi_mutex);
  raw_spin_unlock_irq(&q->pi_state->pi_mutex.wait_lock);
  ret = fixup_pi_state_owner(uaddr, q, owner);
  goto out;
 }





 if (rt_mutex_owner(&q->pi_state->pi_mutex) == current)
  printk(KERN_ERR "fixup_owner: ret = %d pi-mutex: %p "
    "pi-state %p\n", ret,
    q->pi_state->pi_mutex.owner,
    q->pi_state->owner);

out:
 return ret ? ret : locked;
}







static void futex_wait_queue_me(struct futex_hash_bucket *hb, struct futex_q *q,
    struct hrtimer_sleeper *timeout)
{






 set_current_state(TASK_INTERRUPTIBLE);
 queue_me(q, hb);


 if (timeout)
  hrtimer_start_expires(&timeout->timer, HRTIMER_MODE_ABS);





 if (likely(!plist_node_empty(&q->list))) {





  if (!timeout || timeout->task)
   freezable_schedule();
 }
 __set_current_state(TASK_RUNNING);
}
static int futex_wait_setup(u32 __user *uaddr, u32 val, unsigned int flags,
      struct futex_q *q, struct futex_hash_bucket **hb)
{
 u32 uval;
 int ret;
retry:
 ret = get_futex_key(uaddr, flags & FLAGS_SHARED, &q->key, VERIFY_READ);
 if (unlikely(ret != 0))
  return ret;

retry_private:
 *hb = queue_lock(q);

 ret = get_futex_value_locked(&uval, uaddr);

 if (ret) {
  queue_unlock(*hb);

  ret = get_user(uval, uaddr);
  if (ret)
   goto out;

  if (!(flags & FLAGS_SHARED))
   goto retry_private;

  put_futex_key(&q->key);
  goto retry;
 }

 if (uval != val) {
  queue_unlock(*hb);
  ret = -EWOULDBLOCK;
 }

out:
 if (ret)
  put_futex_key(&q->key);
 return ret;
}

static int futex_wait(u32 __user *uaddr, unsigned int flags, u32 val,
        ktime_t *abs_time, u32 bitset)
{
 struct hrtimer_sleeper timeout, *to = NULL;
 struct restart_block *restart;
 struct futex_hash_bucket *hb;
 struct futex_q q = futex_q_init;
 int ret;

 if (!bitset)
  return -EINVAL;
 q.bitset = bitset;

 if (abs_time) {
  to = &timeout;

  hrtimer_init_on_stack(&to->timer, (flags & FLAGS_CLOCKRT) ?
          CLOCK_REALTIME : CLOCK_MONOTONIC,
          HRTIMER_MODE_ABS);
  hrtimer_init_sleeper(to, current);
  hrtimer_set_expires_range_ns(&to->timer, *abs_time,
          current->timer_slack_ns);
 }

retry:




 ret = futex_wait_setup(uaddr, val, flags, &q, &hb);
 if (ret)
  goto out;


 futex_wait_queue_me(hb, &q, to);


 ret = 0;

 if (!unqueue_me(&q))
  goto out;
 ret = -ETIMEDOUT;
 if (to && !to->task)
  goto out;





 if (!signal_pending(current))
  goto retry;

 ret = -ERESTARTSYS;
 if (!abs_time)
  goto out;

 restart = &current->restart_block;
 restart->fn = futex_wait_restart;
 restart->futex.uaddr = uaddr;
 restart->futex.val = val;
 restart->futex.time = abs_time->tv64;
 restart->futex.bitset = bitset;
 restart->futex.flags = flags | FLAGS_HAS_TIMEOUT;

 ret = -ERESTART_RESTARTBLOCK;

out:
 if (to) {
  hrtimer_cancel(&to->timer);
  destroy_hrtimer_on_stack(&to->timer);
 }
 return ret;
}


static long futex_wait_restart(struct restart_block *restart)
{
 u32 __user *uaddr = restart->futex.uaddr;
 ktime_t t, *tp = NULL;

 if (restart->futex.flags & FLAGS_HAS_TIMEOUT) {
  t.tv64 = restart->futex.time;
  tp = &t;
 }
 restart->fn = do_no_restart_syscall;

 return (long)futex_wait(uaddr, restart->futex.flags,
    restart->futex.val, tp, restart->futex.bitset);
}
static int futex_lock_pi(u32 __user *uaddr, unsigned int flags,
    ktime_t *time, int trylock)
{
 struct hrtimer_sleeper timeout, *to = NULL;
 struct futex_hash_bucket *hb;
 struct futex_q q = futex_q_init;
 int res, ret;

 if (refill_pi_state_cache())
  return -ENOMEM;

 if (time) {
  to = &timeout;
  hrtimer_init_on_stack(&to->timer, CLOCK_REALTIME,
          HRTIMER_MODE_ABS);
  hrtimer_init_sleeper(to, current);
  hrtimer_set_expires(&to->timer, *time);
 }

retry:
 ret = get_futex_key(uaddr, flags & FLAGS_SHARED, &q.key, VERIFY_WRITE);
 if (unlikely(ret != 0))
  goto out;

retry_private:
 hb = queue_lock(&q);

 ret = futex_lock_pi_atomic(uaddr, hb, &q.key, &q.pi_state, current, 0);
 if (unlikely(ret)) {




  switch (ret) {
  case 1:

   ret = 0;
   goto out_unlock_put_key;
  case -EFAULT:
   goto uaddr_faulted;
  case -EAGAIN:






   queue_unlock(hb);
   put_futex_key(&q.key);
   cond_resched();
   goto retry;
  default:
   goto out_unlock_put_key;
  }
 }




 queue_me(&q, hb);

 WARN_ON(!q.pi_state);



 if (!trylock) {
  ret = rt_mutex_timed_futex_lock(&q.pi_state->pi_mutex, to);
 } else {
  ret = rt_mutex_trylock(&q.pi_state->pi_mutex);

  ret = ret ? 0 : -EWOULDBLOCK;
 }

 spin_lock(q.lock_ptr);




 res = fixup_owner(uaddr, &q, !ret);




 if (res)
  ret = (res < 0) ? res : 0;





 if (ret && (rt_mutex_owner(&q.pi_state->pi_mutex) == current))
  rt_mutex_unlock(&q.pi_state->pi_mutex);


 unqueue_me_pi(&q);

 goto out_put_key;

out_unlock_put_key:
 queue_unlock(hb);

out_put_key:
 put_futex_key(&q.key);
out:
 if (to)
  destroy_hrtimer_on_stack(&to->timer);
 return ret != -EINTR ? ret : -ERESTARTNOINTR;

uaddr_faulted:
 queue_unlock(hb);

 ret = fault_in_user_writeable(uaddr);
 if (ret)
  goto out_put_key;

 if (!(flags & FLAGS_SHARED))
  goto retry_private;

 put_futex_key(&q.key);
 goto retry;
}






static int futex_unlock_pi(u32 __user *uaddr, unsigned int flags)
{
 u32 uninitialized_var(curval), uval, vpid = task_pid_vnr(current);
 union futex_key key = FUTEX_KEY_INIT;
 struct futex_hash_bucket *hb;
 struct futex_q *match;
 int ret;

retry:
 if (get_user(uval, uaddr))
  return -EFAULT;



 if ((uval & FUTEX_TID_MASK) != vpid)
  return -EPERM;

 ret = get_futex_key(uaddr, flags & FLAGS_SHARED, &key, VERIFY_WRITE);
 if (ret)
  return ret;

 hb = hash_futex(&key);
 spin_lock(&hb->lock);






 match = futex_top_waiter(hb, &key);
 if (match) {
  ret = wake_futex_pi(uaddr, uval, match, hb);




  if (!ret)
   goto out_putkey;




  if (ret == -EFAULT)
   goto pi_faulted;




  if (ret == -EAGAIN) {
   spin_unlock(&hb->lock);
   put_futex_key(&key);
   goto retry;
  }




  goto out_unlock;
 }
 if (cmpxchg_futex_value_locked(&curval, uaddr, uval, 0))
  goto pi_faulted;




 ret = (curval == uval) ? 0 : -EAGAIN;

out_unlock:
 spin_unlock(&hb->lock);
out_putkey:
 put_futex_key(&key);
 return ret;

pi_faulted:
 spin_unlock(&hb->lock);
 put_futex_key(&key);

 ret = fault_in_user_writeable(uaddr);
 if (!ret)
  goto retry;

 return ret;
}
static inline
int handle_early_requeue_pi_wakeup(struct futex_hash_bucket *hb,
       struct futex_q *q, union futex_key *key2,
       struct hrtimer_sleeper *timeout)
{
 int ret = 0;
 if (!match_futex(&q->key, key2)) {
  WARN_ON(q->lock_ptr && (&hb->lock != q->lock_ptr));




  plist_del(&q->list, &hb->chain);
  hb_waiters_dec(hb);


  ret = -EWOULDBLOCK;
  if (timeout && !timeout->task)
   ret = -ETIMEDOUT;
  else if (signal_pending(current))
   ret = -ERESTARTNOINTR;
 }
 return ret;
}
static int futex_wait_requeue_pi(u32 __user *uaddr, unsigned int flags,
     u32 val, ktime_t *abs_time, u32 bitset,
     u32 __user *uaddr2)
{
 struct hrtimer_sleeper timeout, *to = NULL;
 struct rt_mutex_waiter rt_waiter;
 struct rt_mutex *pi_mutex = NULL;
 struct futex_hash_bucket *hb;
 union futex_key key2 = FUTEX_KEY_INIT;
 struct futex_q q = futex_q_init;
 int res, ret;

 if (uaddr == uaddr2)
  return -EINVAL;

 if (!bitset)
  return -EINVAL;

 if (abs_time) {
  to = &timeout;
  hrtimer_init_on_stack(&to->timer, (flags & FLAGS_CLOCKRT) ?
          CLOCK_REALTIME : CLOCK_MONOTONIC,
          HRTIMER_MODE_ABS);
  hrtimer_init_sleeper(to, current);
  hrtimer_set_expires_range_ns(&to->timer, *abs_time,
          current->timer_slack_ns);
 }





 debug_rt_mutex_init_waiter(&rt_waiter);
 RB_CLEAR_NODE(&rt_waiter.pi_tree_entry);
 RB_CLEAR_NODE(&rt_waiter.tree_entry);
 rt_waiter.task = NULL;

 ret = get_futex_key(uaddr2, flags & FLAGS_SHARED, &key2, VERIFY_WRITE);
 if (unlikely(ret != 0))
  goto out;

 q.bitset = bitset;
 q.rt_waiter = &rt_waiter;
 q.requeue_pi_key = &key2;





 ret = futex_wait_setup(uaddr, val, flags, &q, &hb);
 if (ret)
  goto out_key2;





 if (match_futex(&q.key, &key2)) {
  queue_unlock(hb);
  ret = -EINVAL;
  goto out_put_keys;
 }


 futex_wait_queue_me(hb, &q, to);

 spin_lock(&hb->lock);
 ret = handle_early_requeue_pi_wakeup(hb, &q, &key2, to);
 spin_unlock(&hb->lock);
 if (ret)
  goto out_put_keys;
 if (!q.rt_waiter) {




  if (q.pi_state && (q.pi_state->owner != current)) {
   spin_lock(q.lock_ptr);
   ret = fixup_pi_state_owner(uaddr2, &q, current);




   put_pi_state(q.pi_state);
   spin_unlock(q.lock_ptr);
  }
 } else {





  WARN_ON(!q.pi_state);
  pi_mutex = &q.pi_state->pi_mutex;
  ret = rt_mutex_finish_proxy_lock(pi_mutex, to, &rt_waiter);
  debug_rt_mutex_free_waiter(&rt_waiter);

  spin_lock(q.lock_ptr);




  res = fixup_owner(uaddr2, &q, !ret);




  if (res)
   ret = (res < 0) ? res : 0;


  unqueue_me_pi(&q);
 }





 if (ret == -EFAULT) {
  if (pi_mutex && rt_mutex_owner(pi_mutex) == current)
   rt_mutex_unlock(pi_mutex);
 } else if (ret == -EINTR) {







  ret = -EWOULDBLOCK;
 }

out_put_keys:
 put_futex_key(&q.key);
out_key2:
 put_futex_key(&key2);

out:
 if (to) {
  hrtimer_cancel(&to->timer);
  destroy_hrtimer_on_stack(&to->timer);
 }
 return ret;
}
SYSCALL_DEFINE2(set_robust_list, struct robust_list_head __user *, head,
  size_t, len)
{
 if (!futex_cmpxchg_enabled)
  return -ENOSYS;



 if (unlikely(len != sizeof(*head)))
  return -EINVAL;

 current->robust_list = head;

 return 0;
}







SYSCALL_DEFINE3(get_robust_list, int, pid,
  struct robust_list_head __user * __user *, head_ptr,
  size_t __user *, len_ptr)
{
 struct robust_list_head __user *head;
 unsigned long ret;
 struct task_struct *p;

 if (!futex_cmpxchg_enabled)
  return -ENOSYS;

 rcu_read_lock();

 ret = -ESRCH;
 if (!pid)
  p = current;
 else {
  p = find_task_by_vpid(pid);
  if (!p)
   goto err_unlock;
 }

 ret = -EPERM;
 if (!ptrace_may_access(p, PTRACE_MODE_READ_REALCREDS))
  goto err_unlock;

 head = p->robust_list;
 rcu_read_unlock();

 if (put_user(sizeof(*head), len_ptr))
  return -EFAULT;
 return put_user(head, head_ptr);

err_unlock:
 rcu_read_unlock();

 return ret;
}





int handle_futex_death(u32 __user *uaddr, struct task_struct *curr, int pi)
{
 u32 uval, uninitialized_var(nval), mval;

retry:
 if (get_user(uval, uaddr))
  return -1;

 if ((uval & FUTEX_TID_MASK) == task_pid_vnr(curr)) {
  mval = (uval & FUTEX_WAITERS) | FUTEX_OWNER_DIED;
  if (cmpxchg_futex_value_locked(&nval, uaddr, uval, mval)) {
   if (fault_in_user_writeable(uaddr))
    return -1;
   goto retry;
  }
  if (nval != uval)
   goto retry;





  if (!pi && (uval & FUTEX_WAITERS))
   futex_wake(uaddr, 1, 1, FUTEX_BITSET_MATCH_ANY);
 }
 return 0;
}




static inline int fetch_robust_entry(struct robust_list __user **entry,
         struct robust_list __user * __user *head,
         unsigned int *pi)
{
 unsigned long uentry;

 if (get_user(uentry, (unsigned long __user *)head))
  return -EFAULT;

 *entry = (void __user *)(uentry & ~1UL);
 *pi = uentry & 1;

 return 0;
}







void exit_robust_list(struct task_struct *curr)
{
 struct robust_list_head __user *head = curr->robust_list;
 struct robust_list __user *entry, *next_entry, *pending;
 unsigned int limit = ROBUST_LIST_LIMIT, pi, pip;
 unsigned int uninitialized_var(next_pi);
 unsigned long futex_offset;
 int rc;

 if (!futex_cmpxchg_enabled)
  return;





 if (fetch_robust_entry(&entry, &head->list.next, &pi))
  return;



 if (get_user(futex_offset, &head->futex_offset))
  return;




 if (fetch_robust_entry(&pending, &head->list_op_pending, &pip))
  return;

 next_entry = NULL;
 while (entry != &head->list) {




  rc = fetch_robust_entry(&next_entry, &entry->next, &next_pi);




  if (entry != pending)
   if (handle_futex_death((void __user *)entry + futex_offset,
      curr, pi))
    return;
  if (rc)
   return;
  entry = next_entry;
  pi = next_pi;



  if (!--limit)
   break;

  cond_resched();
 }

 if (pending)
  handle_futex_death((void __user *)pending + futex_offset,
       curr, pip);
}

long do_futex(u32 __user *uaddr, int op, u32 val, ktime_t *timeout,
  u32 __user *uaddr2, u32 val2, u32 val3)
{
 int cmd = op & FUTEX_CMD_MASK;
 unsigned int flags = 0;

 if (!(op & FUTEX_PRIVATE_FLAG))
  flags |= FLAGS_SHARED;

 if (op & FUTEX_CLOCK_REALTIME) {
  flags |= FLAGS_CLOCKRT;
  if (cmd != FUTEX_WAIT && cmd != FUTEX_WAIT_BITSET && \
      cmd != FUTEX_WAIT_REQUEUE_PI)
   return -ENOSYS;
 }

 switch (cmd) {
 case FUTEX_LOCK_PI:
 case FUTEX_UNLOCK_PI:
 case FUTEX_TRYLOCK_PI:
 case FUTEX_WAIT_REQUEUE_PI:
 case FUTEX_CMP_REQUEUE_PI:
  if (!futex_cmpxchg_enabled)
   return -ENOSYS;
 }

 switch (cmd) {
 case FUTEX_WAIT:
  val3 = FUTEX_BITSET_MATCH_ANY;
 case FUTEX_WAIT_BITSET:
  return futex_wait(uaddr, flags, val, timeout, val3);
 case FUTEX_WAKE:
  val3 = FUTEX_BITSET_MATCH_ANY;
 case FUTEX_WAKE_BITSET:
  return futex_wake(uaddr, flags, val, val3);
 case FUTEX_REQUEUE:
  return futex_requeue(uaddr, flags, uaddr2, val, val2, NULL, 0);
 case FUTEX_CMP_REQUEUE:
  return futex_requeue(uaddr, flags, uaddr2, val, val2, &val3, 0);
 case FUTEX_WAKE_OP:
  return futex_wake_op(uaddr, flags, uaddr2, val, val2, val3);
 case FUTEX_LOCK_PI:
  return futex_lock_pi(uaddr, flags, timeout, 0);
 case FUTEX_UNLOCK_PI:
  return futex_unlock_pi(uaddr, flags);
 case FUTEX_TRYLOCK_PI:
  return futex_lock_pi(uaddr, flags, NULL, 1);
 case FUTEX_WAIT_REQUEUE_PI:
  val3 = FUTEX_BITSET_MATCH_ANY;
  return futex_wait_requeue_pi(uaddr, flags, val, timeout, val3,
          uaddr2);
 case FUTEX_CMP_REQUEUE_PI:
  return futex_requeue(uaddr, flags, uaddr2, val, val2, &val3, 1);
 }
 return -ENOSYS;
}


SYSCALL_DEFINE6(futex, u32 __user *, uaddr, int, op, u32, val,
  struct timespec __user *, utime, u32 __user *, uaddr2,
  u32, val3)
{
 struct timespec ts;
 ktime_t t, *tp = NULL;
 u32 val2 = 0;
 int cmd = op & FUTEX_CMD_MASK;

 if (utime && (cmd == FUTEX_WAIT || cmd == FUTEX_LOCK_PI ||
        cmd == FUTEX_WAIT_BITSET ||
        cmd == FUTEX_WAIT_REQUEUE_PI)) {
  if (unlikely(should_fail_futex(!(op & FUTEX_PRIVATE_FLAG))))
   return -EFAULT;
  if (copy_from_user(&ts, utime, sizeof(ts)) != 0)
   return -EFAULT;
  if (!timespec_valid(&ts))
   return -EINVAL;

  t = timespec_to_ktime(ts);
  if (cmd == FUTEX_WAIT)
   t = ktime_add_safe(ktime_get(), t);
  tp = &t;
 }




 if (cmd == FUTEX_REQUEUE || cmd == FUTEX_CMP_REQUEUE ||
     cmd == FUTEX_CMP_REQUEUE_PI || cmd == FUTEX_WAKE_OP)
  val2 = (u32) (unsigned long) utime;

 return do_futex(uaddr, op, val, tp, uaddr2, val2, val3);
}

static void __init futex_detect_cmpxchg(void)
{
 u32 curval;
 if (cmpxchg_futex_value_locked(&curval, NULL, 0, 0) == -EFAULT)
  futex_cmpxchg_enabled = 1;
}

static int __init futex_init(void)
{
 unsigned int futex_shift;
 unsigned long i;

 futex_hashsize = 16;
 futex_hashsize = roundup_pow_of_two(256 * num_possible_cpus());

 futex_queues = alloc_large_system_hash("futex", sizeof(*futex_queues),
            futex_hashsize, 0,
            futex_hashsize < 256 ? HASH_SMALL : 0,
            &futex_shift, NULL,
            futex_hashsize, futex_hashsize);
 futex_hashsize = 1UL << futex_shift;

 futex_detect_cmpxchg();

 for (i = 0; i < futex_hashsize; i++) {
  atomic_set(&futex_queues[i].waiters, 0);
  plist_head_init(&futex_queues[i].chain);
  spin_lock_init(&futex_queues[i].lock);
 }

 return 0;
}
__initcall(futex_init);






static inline int
fetch_robust_entry(compat_uptr_t *uentry, struct robust_list __user **entry,
     compat_uptr_t __user *head, unsigned int *pi)
{
 if (get_user(*uentry, head))
  return -EFAULT;

 *entry = compat_ptr((*uentry) & ~1);
 *pi = (unsigned int)(*uentry) & 1;

 return 0;
}

static void __user *futex_uaddr(struct robust_list __user *entry,
    compat_long_t futex_offset)
{
 compat_uptr_t base = ptr_to_compat(entry);
 void __user *uaddr = compat_ptr(base + futex_offset);

 return uaddr;
}







void compat_exit_robust_list(struct task_struct *curr)
{
 struct compat_robust_list_head __user *head = curr->compat_robust_list;
 struct robust_list __user *entry, *next_entry, *pending;
 unsigned int limit = ROBUST_LIST_LIMIT, pi, pip;
 unsigned int uninitialized_var(next_pi);
 compat_uptr_t uentry, next_uentry, upending;
 compat_long_t futex_offset;
 int rc;

 if (!futex_cmpxchg_enabled)
  return;





 if (fetch_robust_entry(&uentry, &entry, &head->list.next, &pi))
  return;



 if (get_user(futex_offset, &head->futex_offset))
  return;




 if (fetch_robust_entry(&upending, &pending,
          &head->list_op_pending, &pip))
  return;

 next_entry = NULL;
 while (entry != (struct robust_list __user *) &head->list) {




  rc = fetch_robust_entry(&next_uentry, &next_entry,
   (compat_uptr_t __user *)&entry->next, &next_pi);




  if (entry != pending) {
   void __user *uaddr = futex_uaddr(entry, futex_offset);

   if (handle_futex_death(uaddr, curr, pi))
    return;
  }
  if (rc)
   return;
  uentry = next_uentry;
  entry = next_entry;
  pi = next_pi;



  if (!--limit)
   break;

  cond_resched();
 }
 if (pending) {
  void __user *uaddr = futex_uaddr(pending, futex_offset);

  handle_futex_death(uaddr, curr, pip);
 }
}

COMPAT_SYSCALL_DEFINE2(set_robust_list,
  struct compat_robust_list_head __user *, head,
  compat_size_t, len)
{
 if (!futex_cmpxchg_enabled)
  return -ENOSYS;

 if (unlikely(len != sizeof(*head)))
  return -EINVAL;

 current->compat_robust_list = head;

 return 0;
}

COMPAT_SYSCALL_DEFINE3(get_robust_list, int, pid,
   compat_uptr_t __user *, head_ptr,
   compat_size_t __user *, len_ptr)
{
 struct compat_robust_list_head __user *head;
 unsigned long ret;
 struct task_struct *p;

 if (!futex_cmpxchg_enabled)
  return -ENOSYS;

 rcu_read_lock();

 ret = -ESRCH;
 if (!pid)
  p = current;
 else {
  p = find_task_by_vpid(pid);
  if (!p)
   goto err_unlock;
 }

 ret = -EPERM;
 if (!ptrace_may_access(p, PTRACE_MODE_READ_REALCREDS))
  goto err_unlock;

 head = p->compat_robust_list;
 rcu_read_unlock();

 if (put_user(sizeof(*head), len_ptr))
  return -EFAULT;
 return put_user(ptr_to_compat(head), head_ptr);

err_unlock:
 rcu_read_unlock();

 return ret;
}

COMPAT_SYSCALL_DEFINE6(futex, u32 __user *, uaddr, int, op, u32, val,
  struct compat_timespec __user *, utime, u32 __user *, uaddr2,
  u32, val3)
{
 struct timespec ts;
 ktime_t t, *tp = NULL;
 int val2 = 0;
 int cmd = op & FUTEX_CMD_MASK;

 if (utime && (cmd == FUTEX_WAIT || cmd == FUTEX_LOCK_PI ||
        cmd == FUTEX_WAIT_BITSET ||
        cmd == FUTEX_WAIT_REQUEUE_PI)) {
  if (compat_get_timespec(&ts, utime))
   return -EFAULT;
  if (!timespec_valid(&ts))
   return -EINVAL;

  t = timespec_to_ktime(ts);
  if (cmd == FUTEX_WAIT)
   t = ktime_add_safe(ktime_get(), t);
  tp = &t;
 }
 if (cmd == FUTEX_REQUEUE || cmd == FUTEX_CMP_REQUEUE ||
     cmd == FUTEX_CMP_REQUEUE_PI || cmd == FUTEX_WAKE_OP)
  val2 = (int) (unsigned long) utime;

 return do_futex(uaddr, op, val, tp, uaddr2, val2, val3);
}



static char remcom_in_buffer[BUFMAX];
static char remcom_out_buffer[BUFMAX];
static int gdbstub_use_prev_in_buf;
static int gdbstub_prev_in_buf_pos;


static unsigned long gdb_regs[(NUMREGBYTES +
     sizeof(unsigned long) - 1) /
     sizeof(unsigned long)];





static int gdbstub_read_wait(void)
{
 int ret = -1;
 int i;

 if (unlikely(gdbstub_use_prev_in_buf)) {
  if (gdbstub_prev_in_buf_pos < gdbstub_use_prev_in_buf)
   return remcom_in_buffer[gdbstub_prev_in_buf_pos++];
  else
   gdbstub_use_prev_in_buf = 0;
 }


 while (ret < 0)
  for (i = 0; kdb_poll_funcs[i] != NULL; i++) {
   ret = kdb_poll_funcs[i]();
   if (ret > 0)
    break;
  }
 return ret;
}
static int gdbstub_read_wait(void)
{
 int ret = dbg_io_ops->read_char();
 while (ret == NO_POLL_CHAR)
  ret = dbg_io_ops->read_char();
 return ret;
}

static void get_packet(char *buffer)
{
 unsigned char checksum;
 unsigned char xmitcsum;
 int count;
 char ch;

 do {




  while ((ch = (gdbstub_read_wait())) != '$')
                ;

  kgdb_connected = 1;
  checksum = 0;
  xmitcsum = -1;

  count = 0;




  while (count < (BUFMAX - 1)) {
   ch = gdbstub_read_wait();
   if (ch == '#')
    break;
   checksum = checksum + ch;
   buffer[count] = ch;
   count = count + 1;
  }

  if (ch == '#') {
   xmitcsum = hex_to_bin(gdbstub_read_wait()) << 4;
   xmitcsum += hex_to_bin(gdbstub_read_wait());

   if (checksum != xmitcsum)

    dbg_io_ops->write_char('-');
   else

    dbg_io_ops->write_char('+');
   if (dbg_io_ops->flush)
    dbg_io_ops->flush();
  }
  buffer[count] = 0;
 } while (checksum != xmitcsum);
}





static void put_packet(char *buffer)
{
 unsigned char checksum;
 int count;
 char ch;




 while (1) {
  dbg_io_ops->write_char('$');
  checksum = 0;
  count = 0;

  while ((ch = buffer[count])) {
   dbg_io_ops->write_char(ch);
   checksum += ch;
   count++;
  }

  dbg_io_ops->write_char('#');
  dbg_io_ops->write_char(hex_asc_hi(checksum));
  dbg_io_ops->write_char(hex_asc_lo(checksum));
  if (dbg_io_ops->flush)
   dbg_io_ops->flush();


  ch = gdbstub_read_wait();

  if (ch == 3)
   ch = gdbstub_read_wait();


  if (ch == '+')
   return;







  if (ch == '$') {
   dbg_io_ops->write_char('-');
   if (dbg_io_ops->flush)
    dbg_io_ops->flush();
   return;
  }
 }
}

static char gdbmsgbuf[BUFMAX + 1];

void gdbstub_msg_write(const char *s, int len)
{
 char *bufptr;
 int wcount;
 int i;

 if (len == 0)
  len = strlen(s);


 gdbmsgbuf[0] = 'O';


 while (len > 0) {
  bufptr = gdbmsgbuf + 1;


  if ((len << 1) > (BUFMAX - 2))
   wcount = (BUFMAX - 2) >> 1;
  else
   wcount = len;


  for (i = 0; i < wcount; i++)
   bufptr = hex_byte_pack(bufptr, s[i]);
  *bufptr = '\0';


  s += wcount;
  len -= wcount;


  put_packet(gdbmsgbuf);
 }
}






char *kgdb_mem2hex(char *mem, char *buf, int count)
{
 char *tmp;
 int err;





 tmp = buf + count;

 err = probe_kernel_read(tmp, mem, count);
 if (err)
  return NULL;
 while (count > 0) {
  buf = hex_byte_pack(buf, *tmp);
  tmp++;
  count--;
 }
 *buf = 0;

 return buf;
}






int kgdb_hex2mem(char *buf, char *mem, int count)
{
 char *tmp_raw;
 char *tmp_hex;





 tmp_raw = buf + count * 2;

 tmp_hex = tmp_raw - 1;
 while (tmp_hex >= buf) {
  tmp_raw--;
  *tmp_raw = hex_to_bin(*tmp_hex--);
  *tmp_raw |= hex_to_bin(*tmp_hex--) << 4;
 }

 return probe_kernel_write(mem, tmp_raw, count);
}





int kgdb_hex2long(char **ptr, unsigned long *long_val)
{
 int hex_val;
 int num = 0;
 int negate = 0;

 *long_val = 0;

 if (**ptr == '-') {
  negate = 1;
  (*ptr)++;
 }
 while (**ptr) {
  hex_val = hex_to_bin(**ptr);
  if (hex_val < 0)
   break;

  *long_val = (*long_val << 4) | hex_val;
  num++;
  (*ptr)++;
 }

 if (negate)
  *long_val = -*long_val;

 return num;
}






static int kgdb_ebin2mem(char *buf, char *mem, int count)
{
 int size = 0;
 char *c = buf;

 while (count-- > 0) {
  c[size] = *buf++;
  if (c[size] == 0x7d)
   c[size] = *buf++ ^ 0x20;
  size++;
 }

 return probe_kernel_write(mem, c, size);
}

void pt_regs_to_gdb_regs(unsigned long *gdb_regs, struct pt_regs *regs)
{
 int i;
 int idx = 0;
 char *ptr = (char *)gdb_regs;

 for (i = 0; i < DBG_MAX_REG_NUM; i++) {
  dbg_get_reg(i, ptr + idx, regs);
  idx += dbg_reg_def[i].size;
 }
}

void gdb_regs_to_pt_regs(unsigned long *gdb_regs, struct pt_regs *regs)
{
 int i;
 int idx = 0;
 char *ptr = (char *)gdb_regs;

 for (i = 0; i < DBG_MAX_REG_NUM; i++) {
  dbg_set_reg(i, ptr + idx, regs);
  idx += dbg_reg_def[i].size;
 }
}


static int write_mem_msg(int binary)
{
 char *ptr = &remcom_in_buffer[1];
 unsigned long addr;
 unsigned long length;
 int err;

 if (kgdb_hex2long(&ptr, &addr) > 0 && *(ptr++) == ',' &&
     kgdb_hex2long(&ptr, &length) > 0 && *(ptr++) == ':') {
  if (binary)
   err = kgdb_ebin2mem(ptr, (char *)addr, length);
  else
   err = kgdb_hex2mem(ptr, (char *)addr, length);
  if (err)
   return err;
  if (CACHE_FLUSH_IS_SAFE)
   flush_icache_range(addr, addr + length);
  return 0;
 }

 return -EINVAL;
}

static void error_packet(char *pkt, int error)
{
 error = -error;
 pkt[0] = 'E';
 pkt[1] = hex_asc[(error / 10)];
 pkt[2] = hex_asc[(error % 10)];
 pkt[3] = '\0';
}








static char *pack_threadid(char *pkt, unsigned char *id)
{
 unsigned char *limit;
 int lzero = 1;

 limit = id + (BUF_THREAD_ID_SIZE / 2);
 while (id < limit) {
  if (!lzero || *id != 0) {
   pkt = hex_byte_pack(pkt, *id);
   lzero = 0;
  }
  id++;
 }

 if (lzero)
  pkt = hex_byte_pack(pkt, 0);

 return pkt;
}

static void int_to_threadref(unsigned char *id, int value)
{
 put_unaligned_be32(value, id);
}

static struct task_struct *getthread(struct pt_regs *regs, int tid)
{



 if (tid == 0 || tid == -1)
  tid = -atomic_read(&kgdb_active) - 2;
 if (tid < -1 && tid > -NR_CPUS - 2) {
  if (kgdb_info[-tid - 2].task)
   return kgdb_info[-tid - 2].task;
  else
   return idle_task(-tid - 2);
 }
 if (tid <= 0) {
  printk(KERN_ERR "KGDB: Internal thread select error\n");
  dump_stack();
  return NULL;
 }






 return find_task_by_pid_ns(tid, &init_pid_ns);
}






static inline int shadow_pid(int realpid)
{
 if (realpid)
  return realpid;

 return -raw_smp_processor_id() - 2;
}
static void gdb_cmd_status(struct kgdb_state *ks)
{






 dbg_remove_all_break();

 remcom_out_buffer[0] = 'S';
 hex_byte_pack(&remcom_out_buffer[1], ks->signo);
}

static void gdb_get_regs_helper(struct kgdb_state *ks)
{
 struct task_struct *thread;
 void *local_debuggerinfo;
 int i;

 thread = kgdb_usethread;
 if (!thread) {
  thread = kgdb_info[ks->cpu].task;
  local_debuggerinfo = kgdb_info[ks->cpu].debuggerinfo;
 } else {
  local_debuggerinfo = NULL;
  for_each_online_cpu(i) {






   if (thread == kgdb_info[i].task)
    local_debuggerinfo = kgdb_info[i].debuggerinfo;
  }
 }






 if (local_debuggerinfo) {
  pt_regs_to_gdb_regs(gdb_regs, local_debuggerinfo);
 } else {







  sleeping_thread_to_gdb_regs(gdb_regs, thread);
 }
}


static void gdb_cmd_getregs(struct kgdb_state *ks)
{
 gdb_get_regs_helper(ks);
 kgdb_mem2hex((char *)gdb_regs, remcom_out_buffer, NUMREGBYTES);
}


static void gdb_cmd_setregs(struct kgdb_state *ks)
{
 kgdb_hex2mem(&remcom_in_buffer[1], (char *)gdb_regs, NUMREGBYTES);

 if (kgdb_usethread && kgdb_usethread != current) {
  error_packet(remcom_out_buffer, -EINVAL);
 } else {
  gdb_regs_to_pt_regs(gdb_regs, ks->linux_regs);
  strcpy(remcom_out_buffer, "OK");
 }
}


static void gdb_cmd_memread(struct kgdb_state *ks)
{
 char *ptr = &remcom_in_buffer[1];
 unsigned long length;
 unsigned long addr;
 char *err;

 if (kgdb_hex2long(&ptr, &addr) > 0 && *ptr++ == ',' &&
     kgdb_hex2long(&ptr, &length) > 0) {
  err = kgdb_mem2hex((char *)addr, remcom_out_buffer, length);
  if (!err)
   error_packet(remcom_out_buffer, -EINVAL);
 } else {
  error_packet(remcom_out_buffer, -EINVAL);
 }
}


static void gdb_cmd_memwrite(struct kgdb_state *ks)
{
 int err = write_mem_msg(0);

 if (err)
  error_packet(remcom_out_buffer, err);
 else
  strcpy(remcom_out_buffer, "OK");
}

static char *gdb_hex_reg_helper(int regnum, char *out)
{
 int i;
 int offset = 0;

 for (i = 0; i < regnum; i++)
  offset += dbg_reg_def[i].size;
 return kgdb_mem2hex((char *)gdb_regs + offset, out,
       dbg_reg_def[i].size);
}


static void gdb_cmd_reg_get(struct kgdb_state *ks)
{
 unsigned long regnum;
 char *ptr = &remcom_in_buffer[1];

 kgdb_hex2long(&ptr, &regnum);
 if (regnum >= DBG_MAX_REG_NUM) {
  error_packet(remcom_out_buffer, -EINVAL);
  return;
 }
 gdb_get_regs_helper(ks);
 gdb_hex_reg_helper(regnum, remcom_out_buffer);
}


static void gdb_cmd_reg_set(struct kgdb_state *ks)
{
 unsigned long regnum;
 char *ptr = &remcom_in_buffer[1];
 int i = 0;

 kgdb_hex2long(&ptr, &regnum);
 if (*ptr++ != '=' ||
     !(!kgdb_usethread || kgdb_usethread == current) ||
     !dbg_get_reg(regnum, gdb_regs, ks->linux_regs)) {
  error_packet(remcom_out_buffer, -EINVAL);
  return;
 }
 memset(gdb_regs, 0, sizeof(gdb_regs));
 while (i < sizeof(gdb_regs) * 2)
  if (hex_to_bin(ptr[i]) >= 0)
   i++;
  else
   break;
 i = i / 2;
 kgdb_hex2mem(ptr, (char *)gdb_regs, i);
 dbg_set_reg(regnum, gdb_regs, ks->linux_regs);
 strcpy(remcom_out_buffer, "OK");
}


static void gdb_cmd_binwrite(struct kgdb_state *ks)
{
 int err = write_mem_msg(1);

 if (err)
  error_packet(remcom_out_buffer, err);
 else
  strcpy(remcom_out_buffer, "OK");
}


static void gdb_cmd_detachkill(struct kgdb_state *ks)
{
 int error;


 if (remcom_in_buffer[0] == 'D') {
  error = dbg_remove_all_break();
  if (error < 0) {
   error_packet(remcom_out_buffer, error);
  } else {
   strcpy(remcom_out_buffer, "OK");
   kgdb_connected = 0;
  }
  put_packet(remcom_out_buffer);
 } else {




  dbg_remove_all_break();
  kgdb_connected = 0;
 }
}


static int gdb_cmd_reboot(struct kgdb_state *ks)
{

 if (strcmp(remcom_in_buffer, "R0") == 0) {
  printk(KERN_CRIT "Executing emergency reboot\n");
  strcpy(remcom_out_buffer, "OK");
  put_packet(remcom_out_buffer);





  machine_emergency_restart();
  kgdb_connected = 0;

  return 1;
 }
 return 0;
}


static void gdb_cmd_query(struct kgdb_state *ks)
{
 struct task_struct *g;
 struct task_struct *p;
 unsigned char thref[BUF_THREAD_ID_SIZE];
 char *ptr;
 int i;
 int cpu;
 int finished = 0;

 switch (remcom_in_buffer[1]) {
 case 's':
 case 'f':
  if (memcmp(remcom_in_buffer + 2, "ThreadInfo", 10))
   break;

  i = 0;
  remcom_out_buffer[0] = 'm';
  ptr = remcom_out_buffer + 1;
  if (remcom_in_buffer[1] == 'f') {

   for_each_online_cpu(cpu) {
    ks->thr_query = 0;
    int_to_threadref(thref, -cpu - 2);
    ptr = pack_threadid(ptr, thref);
    *(ptr++) = ',';
    i++;
   }
  }

  do_each_thread(g, p) {
   if (i >= ks->thr_query && !finished) {
    int_to_threadref(thref, p->pid);
    ptr = pack_threadid(ptr, thref);
    *(ptr++) = ',';
    ks->thr_query++;
    if (ks->thr_query % KGDB_MAX_THREAD_QUERY == 0)
     finished = 1;
   }
   i++;
  } while_each_thread(g, p);

  *(--ptr) = '\0';
  break;

 case 'C':

  strcpy(remcom_out_buffer, "QC");
  ks->threadid = shadow_pid(current->pid);
  int_to_threadref(thref, ks->threadid);
  pack_threadid(remcom_out_buffer + 2, thref);
  break;
 case 'T':
  if (memcmp(remcom_in_buffer + 1, "ThreadExtraInfo,", 16))
   break;

  ks->threadid = 0;
  ptr = remcom_in_buffer + 17;
  kgdb_hex2long(&ptr, &ks->threadid);
  if (!getthread(ks->linux_regs, ks->threadid)) {
   error_packet(remcom_out_buffer, -EINVAL);
   break;
  }
  if ((int)ks->threadid > 0) {
   kgdb_mem2hex(getthread(ks->linux_regs,
     ks->threadid)->comm,
     remcom_out_buffer, 16);
  } else {
   static char tmpstr[23 + BUF_THREAD_ID_SIZE];

   sprintf(tmpstr, "shadowCPU%d",
     (int)(-ks->threadid - 2));
   kgdb_mem2hex(tmpstr, remcom_out_buffer, strlen(tmpstr));
  }
  break;
 case 'R':
  if (strncmp(remcom_in_buffer, "qRcmd,", 6) == 0) {
   int len = strlen(remcom_in_buffer + 6);

   if ((len % 2) != 0) {
    strcpy(remcom_out_buffer, "E01");
    break;
   }
   kgdb_hex2mem(remcom_in_buffer + 6,
         remcom_out_buffer, len);
   len = len / 2;
   remcom_out_buffer[len++] = 0;

   kdb_common_init_state(ks);
   kdb_parse(remcom_out_buffer);
   kdb_common_deinit_state();

   strcpy(remcom_out_buffer, "OK");
  }
  break;
 }
}


static void gdb_cmd_task(struct kgdb_state *ks)
{
 struct task_struct *thread;
 char *ptr;

 switch (remcom_in_buffer[1]) {
 case 'g':
  ptr = &remcom_in_buffer[2];
  kgdb_hex2long(&ptr, &ks->threadid);
  thread = getthread(ks->linux_regs, ks->threadid);
  if (!thread && ks->threadid > 0) {
   error_packet(remcom_out_buffer, -EINVAL);
   break;
  }
  kgdb_usethread = thread;
  ks->kgdb_usethreadid = ks->threadid;
  strcpy(remcom_out_buffer, "OK");
  break;
 case 'c':
  ptr = &remcom_in_buffer[2];
  kgdb_hex2long(&ptr, &ks->threadid);
  if (!ks->threadid) {
   kgdb_contthread = NULL;
  } else {
   thread = getthread(ks->linux_regs, ks->threadid);
   if (!thread && ks->threadid > 0) {
    error_packet(remcom_out_buffer, -EINVAL);
    break;
   }
   kgdb_contthread = thread;
  }
  strcpy(remcom_out_buffer, "OK");
  break;
 }
}


static void gdb_cmd_thread(struct kgdb_state *ks)
{
 char *ptr = &remcom_in_buffer[1];
 struct task_struct *thread;

 kgdb_hex2long(&ptr, &ks->threadid);
 thread = getthread(ks->linux_regs, ks->threadid);
 if (thread)
  strcpy(remcom_out_buffer, "OK");
 else
  error_packet(remcom_out_buffer, -EINVAL);
}


static void gdb_cmd_break(struct kgdb_state *ks)
{




 char *bpt_type = &remcom_in_buffer[1];
 char *ptr = &remcom_in_buffer[2];
 unsigned long addr;
 unsigned long length;
 int error = 0;

 if (arch_kgdb_ops.set_hw_breakpoint && *bpt_type >= '1') {

  if (*bpt_type > '4')
   return;
 } else {
  if (*bpt_type != '0' && *bpt_type != '1')

   return;
 }





 if (*bpt_type == '1' && !(arch_kgdb_ops.flags & KGDB_HW_BREAKPOINT))

  return;

 if (*(ptr++) != ',') {
  error_packet(remcom_out_buffer, -EINVAL);
  return;
 }
 if (!kgdb_hex2long(&ptr, &addr)) {
  error_packet(remcom_out_buffer, -EINVAL);
  return;
 }
 if (*(ptr++) != ',' ||
  !kgdb_hex2long(&ptr, &length)) {
  error_packet(remcom_out_buffer, -EINVAL);
  return;
 }

 if (remcom_in_buffer[0] == 'Z' && *bpt_type == '0')
  error = dbg_set_sw_break(addr);
 else if (remcom_in_buffer[0] == 'z' && *bpt_type == '0')
  error = dbg_remove_sw_break(addr);
 else if (remcom_in_buffer[0] == 'Z')
  error = arch_kgdb_ops.set_hw_breakpoint(addr,
   (int)length, *bpt_type - '0');
 else if (remcom_in_buffer[0] == 'z')
  error = arch_kgdb_ops.remove_hw_breakpoint(addr,
   (int) length, *bpt_type - '0');

 if (error == 0)
  strcpy(remcom_out_buffer, "OK");
 else
  error_packet(remcom_out_buffer, error);
}


static int gdb_cmd_exception_pass(struct kgdb_state *ks)
{



 if (remcom_in_buffer[1] == '0' && remcom_in_buffer[2] == '9') {

  ks->pass_exception = 1;
  remcom_in_buffer[0] = 'c';

 } else if (remcom_in_buffer[1] == '1' && remcom_in_buffer[2] == '5') {

  ks->pass_exception = 1;
  remcom_in_buffer[0] = 'D';
  dbg_remove_all_break();
  kgdb_connected = 0;
  return 1;

 } else {
  gdbstub_msg_write("KGDB only knows signal 9 (pass)"
   " and 15 (pass and disconnect)\n"
   "Executing a continue without signal passing\n", 0);
  remcom_in_buffer[0] = 'c';
 }


 return -1;
}




int gdb_serial_stub(struct kgdb_state *ks)
{
 int error = 0;
 int tmp;


 memset(remcom_out_buffer, 0, sizeof(remcom_out_buffer));
 kgdb_usethread = kgdb_info[ks->cpu].task;
 ks->kgdb_usethreadid = shadow_pid(kgdb_info[ks->cpu].task->pid);
 ks->pass_exception = 0;

 if (kgdb_connected) {
  unsigned char thref[BUF_THREAD_ID_SIZE];
  char *ptr;


  ptr = remcom_out_buffer;
  *ptr++ = 'T';
  ptr = hex_byte_pack(ptr, ks->signo);
  ptr += strlen(strcpy(ptr, "thread:"));
  int_to_threadref(thref, shadow_pid(current->pid));
  ptr = pack_threadid(ptr, thref);
  *ptr++ = ';';
  put_packet(remcom_out_buffer);
 }

 while (1) {
  error = 0;


  memset(remcom_out_buffer, 0, sizeof(remcom_out_buffer));

  get_packet(remcom_in_buffer);

  switch (remcom_in_buffer[0]) {
  case '?':
   gdb_cmd_status(ks);
   break;
  case 'g':
   gdb_cmd_getregs(ks);
   break;
  case 'G':
   gdb_cmd_setregs(ks);
   break;
  case 'm':
   gdb_cmd_memread(ks);
   break;
  case 'M':
   gdb_cmd_memwrite(ks);
   break;
  case 'p':
   gdb_cmd_reg_get(ks);
   break;
  case 'P':
   gdb_cmd_reg_set(ks);
   break;
  case 'X':
   gdb_cmd_binwrite(ks);
   break;



  case 'D':
  case 'k':
   gdb_cmd_detachkill(ks);
   goto default_handle;
  case 'R':
   if (gdb_cmd_reboot(ks))
    goto default_handle;
   break;
  case 'q':
   gdb_cmd_query(ks);
   break;
  case 'H':
   gdb_cmd_task(ks);
   break;
  case 'T':
   gdb_cmd_thread(ks);
   break;
  case 'z':
  case 'Z':
   gdb_cmd_break(ks);
   break;
  case '3':
   if (remcom_in_buffer[1] == '\0') {
    gdb_cmd_detachkill(ks);
    return DBG_PASS_EVENT;
   }
  case 'C':
   tmp = gdb_cmd_exception_pass(ks);
   if (tmp > 0)
    goto default_handle;
   if (tmp == 0)
    break;

  case 'c':
  case 's':
   if (kgdb_contthread && kgdb_contthread != current) {

    error_packet(remcom_out_buffer, -EINVAL);
    break;
   }
   dbg_activate_sw_breakpoints();

  default:
default_handle:
   error = kgdb_arch_handle_exception(ks->ex_vector,
      ks->signo,
      ks->err_code,
      remcom_in_buffer,
      remcom_out_buffer,
      ks->linux_regs);




   if (error >= 0 || remcom_in_buffer[0] == 'D' ||
       remcom_in_buffer[0] == 'k') {
    error = 0;
    goto kgdb_exit;
   }

  }


  put_packet(remcom_out_buffer);
 }

kgdb_exit:
 if (ks->pass_exception)
  error = 1;
 return error;
}

int gdbstub_state(struct kgdb_state *ks, char *cmd)
{
 int error;

 switch (cmd[0]) {
 case 'e':
  error = kgdb_arch_handle_exception(ks->ex_vector,
         ks->signo,
         ks->err_code,
         remcom_in_buffer,
         remcom_out_buffer,
         ks->linux_regs);
  return error;
 case 's':
 case 'c':
  strcpy(remcom_in_buffer, cmd);
  return 0;
 case '$':
  strcpy(remcom_in_buffer, cmd);
  gdbstub_use_prev_in_buf = strlen(remcom_in_buffer);
  gdbstub_prev_in_buf_pos = 0;
  return 0;
 }
 dbg_io_ops->write_char('+');
 put_packet(remcom_out_buffer);
 return 0;
}





void gdbstub_exit(int status)
{
 unsigned char checksum, ch, buffer[3];
 int loop;

 if (!kgdb_connected)
  return;
 kgdb_connected = 0;

 if (!dbg_io_ops || dbg_kdb_mode)
  return;

 buffer[0] = 'W';
 buffer[1] = hex_asc_hi(status);
 buffer[2] = hex_asc_lo(status);

 dbg_io_ops->write_char('$');
 checksum = 0;

 for (loop = 0; loop < 3; loop++) {
  ch = buffer[loop];
  checksum += ch;
  dbg_io_ops->write_char(ch);
 }

 dbg_io_ops->write_char('#');
 dbg_io_ops->write_char(hex_asc_hi(checksum));
 dbg_io_ops->write_char(hex_asc_lo(checksum));


 if (dbg_io_ops->flush)
  dbg_io_ops->flush();
}







static LIST_HEAD(gc_list);
static DEFINE_RAW_SPINLOCK(gc_lock);





void irq_gc_noop(struct irq_data *d)
{
}
void irq_gc_mask_disable_reg(struct irq_data *d)
{
 struct irq_chip_generic *gc = irq_data_get_irq_chip_data(d);
 struct irq_chip_type *ct = irq_data_get_chip_type(d);
 u32 mask = d->mask;

 irq_gc_lock(gc);
 irq_reg_writel(gc, mask, ct->regs.disable);
 *ct->mask_cache &= ~mask;
 irq_gc_unlock(gc);
}
void irq_gc_mask_set_bit(struct irq_data *d)
{
 struct irq_chip_generic *gc = irq_data_get_irq_chip_data(d);
 struct irq_chip_type *ct = irq_data_get_chip_type(d);
 u32 mask = d->mask;

 irq_gc_lock(gc);
 *ct->mask_cache |= mask;
 irq_reg_writel(gc, *ct->mask_cache, ct->regs.mask);
 irq_gc_unlock(gc);
}
EXPORT_SYMBOL_GPL(irq_gc_mask_set_bit);
void irq_gc_mask_clr_bit(struct irq_data *d)
{
 struct irq_chip_generic *gc = irq_data_get_irq_chip_data(d);
 struct irq_chip_type *ct = irq_data_get_chip_type(d);
 u32 mask = d->mask;

 irq_gc_lock(gc);
 *ct->mask_cache &= ~mask;
 irq_reg_writel(gc, *ct->mask_cache, ct->regs.mask);
 irq_gc_unlock(gc);
}
EXPORT_SYMBOL_GPL(irq_gc_mask_clr_bit);
void irq_gc_unmask_enable_reg(struct irq_data *d)
{
 struct irq_chip_generic *gc = irq_data_get_irq_chip_data(d);
 struct irq_chip_type *ct = irq_data_get_chip_type(d);
 u32 mask = d->mask;

 irq_gc_lock(gc);
 irq_reg_writel(gc, mask, ct->regs.enable);
 *ct->mask_cache |= mask;
 irq_gc_unlock(gc);
}





void irq_gc_ack_set_bit(struct irq_data *d)
{
 struct irq_chip_generic *gc = irq_data_get_irq_chip_data(d);
 struct irq_chip_type *ct = irq_data_get_chip_type(d);
 u32 mask = d->mask;

 irq_gc_lock(gc);
 irq_reg_writel(gc, mask, ct->regs.ack);
 irq_gc_unlock(gc);
}
EXPORT_SYMBOL_GPL(irq_gc_ack_set_bit);





void irq_gc_ack_clr_bit(struct irq_data *d)
{
 struct irq_chip_generic *gc = irq_data_get_irq_chip_data(d);
 struct irq_chip_type *ct = irq_data_get_chip_type(d);
 u32 mask = ~d->mask;

 irq_gc_lock(gc);
 irq_reg_writel(gc, mask, ct->regs.ack);
 irq_gc_unlock(gc);
}





void irq_gc_mask_disable_reg_and_ack(struct irq_data *d)
{
 struct irq_chip_generic *gc = irq_data_get_irq_chip_data(d);
 struct irq_chip_type *ct = irq_data_get_chip_type(d);
 u32 mask = d->mask;

 irq_gc_lock(gc);
 irq_reg_writel(gc, mask, ct->regs.mask);
 irq_reg_writel(gc, mask, ct->regs.ack);
 irq_gc_unlock(gc);
}





void irq_gc_eoi(struct irq_data *d)
{
 struct irq_chip_generic *gc = irq_data_get_irq_chip_data(d);
 struct irq_chip_type *ct = irq_data_get_chip_type(d);
 u32 mask = d->mask;

 irq_gc_lock(gc);
 irq_reg_writel(gc, mask, ct->regs.eoi);
 irq_gc_unlock(gc);
}
int irq_gc_set_wake(struct irq_data *d, unsigned int on)
{
 struct irq_chip_generic *gc = irq_data_get_irq_chip_data(d);
 u32 mask = d->mask;

 if (!(mask & gc->wake_enabled))
  return -EINVAL;

 irq_gc_lock(gc);
 if (on)
  gc->wake_active |= mask;
 else
  gc->wake_active &= ~mask;
 irq_gc_unlock(gc);
 return 0;
}

static u32 irq_readl_be(void __iomem *addr)
{
 return ioread32be(addr);
}

static void irq_writel_be(u32 val, void __iomem *addr)
{
 iowrite32be(val, addr);
}

static void
irq_init_generic_chip(struct irq_chip_generic *gc, const char *name,
        int num_ct, unsigned int irq_base,
        void __iomem *reg_base, irq_flow_handler_t handler)
{
 raw_spin_lock_init(&gc->lock);
 gc->num_ct = num_ct;
 gc->irq_base = irq_base;
 gc->reg_base = reg_base;
 gc->chip_types->chip.name = name;
 gc->chip_types->handler = handler;
}
struct irq_chip_generic *
irq_alloc_generic_chip(const char *name, int num_ct, unsigned int irq_base,
         void __iomem *reg_base, irq_flow_handler_t handler)
{
 struct irq_chip_generic *gc;
 unsigned long sz = sizeof(*gc) + num_ct * sizeof(struct irq_chip_type);

 gc = kzalloc(sz, GFP_KERNEL);
 if (gc) {
  irq_init_generic_chip(gc, name, num_ct, irq_base, reg_base,
          handler);
 }
 return gc;
}
EXPORT_SYMBOL_GPL(irq_alloc_generic_chip);

static void
irq_gc_init_mask_cache(struct irq_chip_generic *gc, enum irq_gc_flags flags)
{
 struct irq_chip_type *ct = gc->chip_types;
 u32 *mskptr = &gc->mask_cache, mskreg = ct->regs.mask;
 int i;

 for (i = 0; i < gc->num_ct; i++) {
  if (flags & IRQ_GC_MASK_CACHE_PER_TYPE) {
   mskptr = &ct[i].mask_cache_priv;
   mskreg = ct[i].regs.mask;
  }
  ct[i].mask_cache = mskptr;
  if (flags & IRQ_GC_INIT_MASK_CACHE)
   *mskptr = irq_reg_readl(gc, mskreg);
 }
}
int irq_alloc_domain_generic_chips(struct irq_domain *d, int irqs_per_chip,
       int num_ct, const char *name,
       irq_flow_handler_t handler,
       unsigned int clr, unsigned int set,
       enum irq_gc_flags gcflags)
{
 struct irq_domain_chip_generic *dgc;
 struct irq_chip_generic *gc;
 int numchips, sz, i;
 unsigned long flags;
 void *tmp;

 if (d->gc)
  return -EBUSY;

 numchips = DIV_ROUND_UP(d->revmap_size, irqs_per_chip);
 if (!numchips)
  return -EINVAL;


 sz = sizeof(*dgc) + numchips * sizeof(gc);
 sz += numchips * (sizeof(*gc) + num_ct * sizeof(struct irq_chip_type));

 tmp = dgc = kzalloc(sz, GFP_KERNEL);
 if (!dgc)
  return -ENOMEM;
 dgc->irqs_per_chip = irqs_per_chip;
 dgc->num_chips = numchips;
 dgc->irq_flags_to_set = set;
 dgc->irq_flags_to_clear = clr;
 dgc->gc_flags = gcflags;
 d->gc = dgc;


 tmp += sizeof(*dgc) + numchips * sizeof(gc);
 for (i = 0; i < numchips; i++) {

  dgc->gc[i] = gc = tmp;
  irq_init_generic_chip(gc, name, num_ct, i * irqs_per_chip,
          NULL, handler);

  gc->domain = d;
  if (gcflags & IRQ_GC_BE_IO) {
   gc->reg_readl = &irq_readl_be;
   gc->reg_writel = &irq_writel_be;
  }

  raw_spin_lock_irqsave(&gc_lock, flags);
  list_add_tail(&gc->list, &gc_list);
  raw_spin_unlock_irqrestore(&gc_lock, flags);

  tmp += sizeof(*gc) + num_ct * sizeof(struct irq_chip_type);
 }
 d->name = name;
 return 0;
}
EXPORT_SYMBOL_GPL(irq_alloc_domain_generic_chips);






struct irq_chip_generic *
irq_get_domain_generic_chip(struct irq_domain *d, unsigned int hw_irq)
{
 struct irq_domain_chip_generic *dgc = d->gc;
 int idx;

 if (!dgc)
  return NULL;
 idx = hw_irq / dgc->irqs_per_chip;
 if (idx >= dgc->num_chips)
  return NULL;
 return dgc->gc[idx];
}
EXPORT_SYMBOL_GPL(irq_get_domain_generic_chip);





static struct lock_class_key irq_nested_lock_class;




int irq_map_generic_chip(struct irq_domain *d, unsigned int virq,
    irq_hw_number_t hw_irq)
{
 struct irq_data *data = irq_domain_get_irq_data(d, virq);
 struct irq_domain_chip_generic *dgc = d->gc;
 struct irq_chip_generic *gc;
 struct irq_chip_type *ct;
 struct irq_chip *chip;
 unsigned long flags;
 int idx;

 if (!d->gc)
  return -ENODEV;

 idx = hw_irq / dgc->irqs_per_chip;
 if (idx >= dgc->num_chips)
  return -EINVAL;
 gc = dgc->gc[idx];

 idx = hw_irq % dgc->irqs_per_chip;

 if (test_bit(idx, &gc->unused))
  return -ENOTSUPP;

 if (test_bit(idx, &gc->installed))
  return -EBUSY;

 ct = gc->chip_types;
 chip = &ct->chip;


 if (!gc->installed) {
  raw_spin_lock_irqsave(&gc->lock, flags);
  irq_gc_init_mask_cache(gc, dgc->gc_flags);
  raw_spin_unlock_irqrestore(&gc->lock, flags);
 }


 set_bit(idx, &gc->installed);

 if (dgc->gc_flags & IRQ_GC_INIT_NESTED_LOCK)
  irq_set_lockdep_class(virq, &irq_nested_lock_class);

 if (chip->irq_calc_mask)
  chip->irq_calc_mask(data);
 else
  data->mask = 1 << idx;

 irq_domain_set_info(d, virq, hw_irq, chip, gc, ct->handler, NULL, NULL);
 irq_modify_status(virq, dgc->irq_flags_to_clear, dgc->irq_flags_to_set);
 return 0;
}
EXPORT_SYMBOL_GPL(irq_map_generic_chip);

struct irq_domain_ops irq_generic_chip_ops = {
 .map = irq_map_generic_chip,
 .xlate = irq_domain_xlate_onetwocell,
};
EXPORT_SYMBOL_GPL(irq_generic_chip_ops);
void irq_setup_generic_chip(struct irq_chip_generic *gc, u32 msk,
       enum irq_gc_flags flags, unsigned int clr,
       unsigned int set)
{
 struct irq_chip_type *ct = gc->chip_types;
 struct irq_chip *chip = &ct->chip;
 unsigned int i;

 raw_spin_lock(&gc_lock);
 list_add_tail(&gc->list, &gc_list);
 raw_spin_unlock(&gc_lock);

 irq_gc_init_mask_cache(gc, flags);

 for (i = gc->irq_base; msk; msk >>= 1, i++) {
  if (!(msk & 0x01))
   continue;

  if (flags & IRQ_GC_INIT_NESTED_LOCK)
   irq_set_lockdep_class(i, &irq_nested_lock_class);

  if (!(flags & IRQ_GC_NO_MASK)) {
   struct irq_data *d = irq_get_irq_data(i);

   if (chip->irq_calc_mask)
    chip->irq_calc_mask(d);
   else
    d->mask = 1 << (i - gc->irq_base);
  }
  irq_set_chip_and_handler(i, chip, ct->handler);
  irq_set_chip_data(i, gc);
  irq_modify_status(i, clr, set);
 }
 gc->irq_cnt = i - gc->irq_base;
}
EXPORT_SYMBOL_GPL(irq_setup_generic_chip);
int irq_setup_alt_chip(struct irq_data *d, unsigned int type)
{
 struct irq_chip_generic *gc = irq_data_get_irq_chip_data(d);
 struct irq_chip_type *ct = gc->chip_types;
 unsigned int i;

 for (i = 0; i < gc->num_ct; i++, ct++) {
  if (ct->type & type) {
   d->chip = &ct->chip;
   irq_data_to_desc(d)->handle_irq = ct->handler;
   return 0;
  }
 }
 return -EINVAL;
}
EXPORT_SYMBOL_GPL(irq_setup_alt_chip);
void irq_remove_generic_chip(struct irq_chip_generic *gc, u32 msk,
        unsigned int clr, unsigned int set)
{
 unsigned int i = gc->irq_base;

 raw_spin_lock(&gc_lock);
 list_del(&gc->list);
 raw_spin_unlock(&gc_lock);

 for (; msk; msk >>= 1, i++) {
  if (!(msk & 0x01))
   continue;


  irq_set_handler(i, NULL);
  irq_set_chip(i, &no_irq_chip);
  irq_set_chip_data(i, NULL);
  irq_modify_status(i, clr, set);
 }
}
EXPORT_SYMBOL_GPL(irq_remove_generic_chip);

static struct irq_data *irq_gc_get_irq_data(struct irq_chip_generic *gc)
{
 unsigned int virq;

 if (!gc->domain)
  return irq_get_irq_data(gc->irq_base);





 if (!gc->installed)
  return NULL;

 virq = irq_find_mapping(gc->domain, gc->irq_base + __ffs(gc->installed));
 return virq ? irq_get_irq_data(virq) : NULL;
}

static int irq_gc_suspend(void)
{
 struct irq_chip_generic *gc;

 list_for_each_entry(gc, &gc_list, list) {
  struct irq_chip_type *ct = gc->chip_types;

  if (ct->chip.irq_suspend) {
   struct irq_data *data = irq_gc_get_irq_data(gc);

   if (data)
    ct->chip.irq_suspend(data);
  }

  if (gc->suspend)
   gc->suspend(gc);
 }
 return 0;
}

static void irq_gc_resume(void)
{
 struct irq_chip_generic *gc;

 list_for_each_entry(gc, &gc_list, list) {
  struct irq_chip_type *ct = gc->chip_types;

  if (gc->resume)
   gc->resume(gc);

  if (ct->chip.irq_resume) {
   struct irq_data *data = irq_gc_get_irq_data(gc);

   if (data)
    ct->chip.irq_resume(data);
  }
 }
}

static void irq_gc_shutdown(void)
{
 struct irq_chip_generic *gc;

 list_for_each_entry(gc, &gc_list, list) {
  struct irq_chip_type *ct = gc->chip_types;

  if (ct->chip.irq_pm_shutdown) {
   struct irq_data *data = irq_gc_get_irq_data(gc);

   if (data)
    ct->chip.irq_pm_shutdown(data);
  }
 }
}

static struct syscore_ops irq_gc_syscore_ops = {
 .suspend = irq_gc_suspend,
 .resume = irq_gc_resume,
 .shutdown = irq_gc_shutdown,
};

static int __init irq_gc_init_ops(void)
{
 register_syscore_ops(&irq_gc_syscore_ops);
 return 0;
}
device_initcall(irq_gc_init_ops);




struct group_info *groups_alloc(int gidsetsize)
{
 struct group_info *group_info;
 int nblocks;
 int i;

 nblocks = (gidsetsize + NGROUPS_PER_BLOCK - 1) / NGROUPS_PER_BLOCK;

 nblocks = nblocks ? : 1;
 group_info = kmalloc(sizeof(*group_info) + nblocks*sizeof(gid_t *), GFP_USER);
 if (!group_info)
  return NULL;
 group_info->ngroups = gidsetsize;
 group_info->nblocks = nblocks;
 atomic_set(&group_info->usage, 1);

 if (gidsetsize <= NGROUPS_SMALL)
  group_info->blocks[0] = group_info->small_block;
 else {
  for (i = 0; i < nblocks; i++) {
   kgid_t *b;
   b = (void *)__get_free_page(GFP_USER);
   if (!b)
    goto out_undo_partial_alloc;
   group_info->blocks[i] = b;
  }
 }
 return group_info;

out_undo_partial_alloc:
 while (--i >= 0) {
  free_page((unsigned long)group_info->blocks[i]);
 }
 kfree(group_info);
 return NULL;
}

EXPORT_SYMBOL(groups_alloc);

void groups_free(struct group_info *group_info)
{
 if (group_info->blocks[0] != group_info->small_block) {
  int i;
  for (i = 0; i < group_info->nblocks; i++)
   free_page((unsigned long)group_info->blocks[i]);
 }
 kfree(group_info);
}

EXPORT_SYMBOL(groups_free);


static int groups_to_user(gid_t __user *grouplist,
     const struct group_info *group_info)
{
 struct user_namespace *user_ns = current_user_ns();
 int i;
 unsigned int count = group_info->ngroups;

 for (i = 0; i < count; i++) {
  gid_t gid;
  gid = from_kgid_munged(user_ns, GROUP_AT(group_info, i));
  if (put_user(gid, grouplist+i))
   return -EFAULT;
 }
 return 0;
}


static int groups_from_user(struct group_info *group_info,
    gid_t __user *grouplist)
{
 struct user_namespace *user_ns = current_user_ns();
 int i;
 unsigned int count = group_info->ngroups;

 for (i = 0; i < count; i++) {
  gid_t gid;
  kgid_t kgid;
  if (get_user(gid, grouplist+i))
   return -EFAULT;

  kgid = make_kgid(user_ns, gid);
  if (!gid_valid(kgid))
   return -EINVAL;

  GROUP_AT(group_info, i) = kgid;
 }
 return 0;
}


static void groups_sort(struct group_info *group_info)
{
 int base, max, stride;
 int gidsetsize = group_info->ngroups;

 for (stride = 1; stride < gidsetsize; stride = 3 * stride + 1)
  ;
 stride /= 3;

 while (stride) {
  max = gidsetsize - stride;
  for (base = 0; base < max; base++) {
   int left = base;
   int right = left + stride;
   kgid_t tmp = GROUP_AT(group_info, right);

   while (left >= 0 && gid_gt(GROUP_AT(group_info, left), tmp)) {
    GROUP_AT(group_info, right) =
        GROUP_AT(group_info, left);
    right = left;
    left -= stride;
   }
   GROUP_AT(group_info, right) = tmp;
  }
  stride /= 3;
 }
}


int groups_search(const struct group_info *group_info, kgid_t grp)
{
 unsigned int left, right;

 if (!group_info)
  return 0;

 left = 0;
 right = group_info->ngroups;
 while (left < right) {
  unsigned int mid = (left+right)/2;
  if (gid_gt(grp, GROUP_AT(group_info, mid)))
   left = mid + 1;
  else if (gid_lt(grp, GROUP_AT(group_info, mid)))
   right = mid;
  else
   return 1;
 }
 return 0;
}






void set_groups(struct cred *new, struct group_info *group_info)
{
 put_group_info(new->group_info);
 groups_sort(group_info);
 get_group_info(group_info);
 new->group_info = group_info;
}

EXPORT_SYMBOL(set_groups);
int set_current_groups(struct group_info *group_info)
{
 struct cred *new;

 new = prepare_creds();
 if (!new)
  return -ENOMEM;

 set_groups(new, group_info);
 return commit_creds(new);
}

EXPORT_SYMBOL(set_current_groups);

SYSCALL_DEFINE2(getgroups, int, gidsetsize, gid_t __user *, grouplist)
{
 const struct cred *cred = current_cred();
 int i;

 if (gidsetsize < 0)
  return -EINVAL;


 i = cred->group_info->ngroups;
 if (gidsetsize) {
  if (i > gidsetsize) {
   i = -EINVAL;
   goto out;
  }
  if (groups_to_user(grouplist, cred->group_info)) {
   i = -EFAULT;
   goto out;
  }
 }
out:
 return i;
}

bool may_setgroups(void)
{
 struct user_namespace *user_ns = current_user_ns();

 return ns_capable(user_ns, CAP_SETGID) &&
  userns_may_setgroups(user_ns);
}






SYSCALL_DEFINE2(setgroups, int, gidsetsize, gid_t __user *, grouplist)
{
 struct group_info *group_info;
 int retval;

 if (!may_setgroups())
  return -EPERM;
 if ((unsigned)gidsetsize > NGROUPS_MAX)
  return -EINVAL;

 group_info = groups_alloc(gidsetsize);
 if (!group_info)
  return -ENOMEM;
 retval = groups_from_user(group_info, grouplist);
 if (retval) {
  put_group_info(group_info);
  return retval;
 }

 retval = set_current_groups(group_info);
 put_group_info(group_info);

 return retval;
}




int in_group_p(kgid_t grp)
{
 const struct cred *cred = current_cred();
 int retval = 1;

 if (!gid_eq(grp, cred->fsgid))
  retval = groups_search(cred->group_info, grp);
 return retval;
}

EXPORT_SYMBOL(in_group_p);

int in_egroup_p(kgid_t grp)
{
 const struct cred *cred = current_cred();
 int retval = 1;

 if (!gid_eq(grp, cred->egid))
  retval = groups_search(cred->group_info, grp);
 return retval;
}

EXPORT_SYMBOL(in_egroup_p);









void handle_bad_irq(struct irq_desc *desc)
{
 unsigned int irq = irq_desc_get_irq(desc);

 print_irq_desc(irq, desc);
 kstat_incr_irqs_this_cpu(desc);
 ack_bad_irq(irq);
}
EXPORT_SYMBOL_GPL(handle_bad_irq);




irqreturn_t no_action(int cpl, void *dev_id)
{
 return IRQ_NONE;
}
EXPORT_SYMBOL_GPL(no_action);

static void warn_no_thread(unsigned int irq, struct irqaction *action)
{
 if (test_and_set_bit(IRQTF_WARNED, &action->thread_flags))
  return;

 printk(KERN_WARNING "IRQ %d device %s returned IRQ_WAKE_THREAD "
        "but no thread function available.", irq, action->name);
}

void __irq_wake_thread(struct irq_desc *desc, struct irqaction *action)
{





 if (action->thread->flags & PF_EXITING)
  return;





 if (test_and_set_bit(IRQTF_RUNTHREAD, &action->thread_flags))
  return;
 desc->threads_oneshot |= action->thread_mask;
 atomic_inc(&desc->threads_active);

 wake_up_process(action->thread);
}

irqreturn_t handle_irq_event_percpu(struct irq_desc *desc)
{
 irqreturn_t retval = IRQ_NONE;
 unsigned int flags = 0, irq = desc->irq_data.irq;
 struct irqaction *action;

 for_each_action_of_desc(desc, action) {
  irqreturn_t res;

  trace_irq_handler_entry(irq, action);
  res = action->handler(irq, action->dev_id);
  trace_irq_handler_exit(irq, action, res);

  if (WARN_ONCE(!irqs_disabled(),"irq %u handler %pF enabled interrupts\n",
         irq, action->handler))
   local_irq_disable();

  switch (res) {
  case IRQ_WAKE_THREAD:




   if (unlikely(!action->thread_fn)) {
    warn_no_thread(irq, action);
    break;
   }

   __irq_wake_thread(desc, action);


  case IRQ_HANDLED:
   flags |= action->flags;
   break;

  default:
   break;
  }

  retval |= res;
 }

 add_interrupt_randomness(irq, flags);

 if (!noirqdebug)
  note_interrupt(desc, retval);
 return retval;
}

irqreturn_t handle_irq_event(struct irq_desc *desc)
{
 irqreturn_t ret;

 desc->istate &= ~IRQS_PENDING;
 irqd_set(&desc->irq_data, IRQD_IRQ_INPROGRESS);
 raw_spin_unlock(&desc->lock);

 ret = handle_irq_event_percpu(desc);

 raw_spin_lock(&desc->lock);
 irqd_clear(&desc->irq_data, IRQD_IRQ_INPROGRESS);
 return ret;
}

struct bucket {
 struct hlist_head head;
 raw_spinlock_t lock;
};

struct bpf_htab {
 struct bpf_map map;
 struct bucket *buckets;
 void *elems;
 struct pcpu_freelist freelist;
 atomic_t count;
 u32 n_buckets;
 u32 elem_size;
};


struct htab_elem {
 union {
  struct hlist_node hash_node;
  struct bpf_htab *htab;
  struct pcpu_freelist_node fnode;
 };
 struct rcu_head rcu;
 u32 hash;
 char key[0] __aligned(8);
};

static inline void htab_elem_set_ptr(struct htab_elem *l, u32 key_size,
         void __percpu *pptr)
{
 *(void __percpu **)(l->key + key_size) = pptr;
}

static inline void __percpu *htab_elem_get_ptr(struct htab_elem *l, u32 key_size)
{
 return *(void __percpu **)(l->key + key_size);
}

static struct htab_elem *get_htab_elem(struct bpf_htab *htab, int i)
{
 return (struct htab_elem *) (htab->elems + i * htab->elem_size);
}

static void htab_free_elems(struct bpf_htab *htab)
{
 int i;

 if (htab->map.map_type != BPF_MAP_TYPE_PERCPU_HASH)
  goto free_elems;

 for (i = 0; i < htab->map.max_entries; i++) {
  void __percpu *pptr;

  pptr = htab_elem_get_ptr(get_htab_elem(htab, i),
      htab->map.key_size);
  free_percpu(pptr);
 }
free_elems:
 vfree(htab->elems);
}

static int prealloc_elems_and_freelist(struct bpf_htab *htab)
{
 int err = -ENOMEM, i;

 htab->elems = vzalloc(htab->elem_size * htab->map.max_entries);
 if (!htab->elems)
  return -ENOMEM;

 if (htab->map.map_type != BPF_MAP_TYPE_PERCPU_HASH)
  goto skip_percpu_elems;

 for (i = 0; i < htab->map.max_entries; i++) {
  u32 size = round_up(htab->map.value_size, 8);
  void __percpu *pptr;

  pptr = __alloc_percpu_gfp(size, 8, GFP_USER | __GFP_NOWARN);
  if (!pptr)
   goto free_elems;
  htab_elem_set_ptr(get_htab_elem(htab, i), htab->map.key_size,
      pptr);
 }

skip_percpu_elems:
 err = pcpu_freelist_init(&htab->freelist);
 if (err)
  goto free_elems;

 pcpu_freelist_populate(&htab->freelist, htab->elems, htab->elem_size,
          htab->map.max_entries);
 return 0;

free_elems:
 htab_free_elems(htab);
 return err;
}


static struct bpf_map *htab_map_alloc(union bpf_attr *attr)
{
 bool percpu = attr->map_type == BPF_MAP_TYPE_PERCPU_HASH;
 struct bpf_htab *htab;
 int err, i;
 u64 cost;

 if (attr->map_flags & ~BPF_F_NO_PREALLOC)

  return ERR_PTR(-EINVAL);

 htab = kzalloc(sizeof(*htab), GFP_USER);
 if (!htab)
  return ERR_PTR(-ENOMEM);


 htab->map.map_type = attr->map_type;
 htab->map.key_size = attr->key_size;
 htab->map.value_size = attr->value_size;
 htab->map.max_entries = attr->max_entries;
 htab->map.map_flags = attr->map_flags;




 err = -EINVAL;
 if (htab->map.max_entries == 0 || htab->map.key_size == 0 ||
     htab->map.value_size == 0)
  goto free_htab;


 htab->n_buckets = roundup_pow_of_two(htab->map.max_entries);

 err = -E2BIG;
 if (htab->map.key_size > MAX_BPF_STACK)



  goto free_htab;

 if (htab->map.value_size >= (1 << (KMALLOC_SHIFT_MAX - 1)) -
     MAX_BPF_STACK - sizeof(struct htab_elem))





  goto free_htab;

 if (percpu && round_up(htab->map.value_size, 8) > PCPU_MIN_UNIT_SIZE)

  goto free_htab;

 htab->elem_size = sizeof(struct htab_elem) +
     round_up(htab->map.key_size, 8);
 if (percpu)
  htab->elem_size += sizeof(void *);
 else
  htab->elem_size += round_up(htab->map.value_size, 8);


 if (htab->n_buckets == 0 ||
     htab->n_buckets > U32_MAX / sizeof(struct bucket))
  goto free_htab;

 cost = (u64) htab->n_buckets * sizeof(struct bucket) +
        (u64) htab->elem_size * htab->map.max_entries;

 if (percpu)
  cost += (u64) round_up(htab->map.value_size, 8) *
   num_possible_cpus() * htab->map.max_entries;

 if (cost >= U32_MAX - PAGE_SIZE)

  goto free_htab;

 htab->map.pages = round_up(cost, PAGE_SIZE) >> PAGE_SHIFT;


 err = bpf_map_precharge_memlock(htab->map.pages);
 if (err)
  goto free_htab;

 err = -ENOMEM;
 htab->buckets = kmalloc_array(htab->n_buckets, sizeof(struct bucket),
          GFP_USER | __GFP_NOWARN);

 if (!htab->buckets) {
  htab->buckets = vmalloc(htab->n_buckets * sizeof(struct bucket));
  if (!htab->buckets)
   goto free_htab;
 }

 for (i = 0; i < htab->n_buckets; i++) {
  INIT_HLIST_HEAD(&htab->buckets[i].head);
  raw_spin_lock_init(&htab->buckets[i].lock);
 }

 if (!(attr->map_flags & BPF_F_NO_PREALLOC)) {
  err = prealloc_elems_and_freelist(htab);
  if (err)
   goto free_buckets;
 }

 return &htab->map;

free_buckets:
 kvfree(htab->buckets);
free_htab:
 kfree(htab);
 return ERR_PTR(err);
}

static inline u32 htab_map_hash(const void *key, u32 key_len)
{
 return jhash(key, key_len, 0);
}

static inline struct bucket *__select_bucket(struct bpf_htab *htab, u32 hash)
{
 return &htab->buckets[hash & (htab->n_buckets - 1)];
}

static inline struct hlist_head *select_bucket(struct bpf_htab *htab, u32 hash)
{
 return &__select_bucket(htab, hash)->head;
}

static struct htab_elem *lookup_elem_raw(struct hlist_head *head, u32 hash,
      void *key, u32 key_size)
{
 struct htab_elem *l;

 hlist_for_each_entry_rcu(l, head, hash_node)
  if (l->hash == hash && !memcmp(&l->key, key, key_size))
   return l;

 return NULL;
}


static void *__htab_map_lookup_elem(struct bpf_map *map, void *key)
{
 struct bpf_htab *htab = container_of(map, struct bpf_htab, map);
 struct hlist_head *head;
 struct htab_elem *l;
 u32 hash, key_size;


 WARN_ON_ONCE(!rcu_read_lock_held());

 key_size = map->key_size;

 hash = htab_map_hash(key, key_size);

 head = select_bucket(htab, hash);

 l = lookup_elem_raw(head, hash, key, key_size);

 return l;
}

static void *htab_map_lookup_elem(struct bpf_map *map, void *key)
{
 struct htab_elem *l = __htab_map_lookup_elem(map, key);

 if (l)
  return l->key + round_up(map->key_size, 8);

 return NULL;
}


static int htab_map_get_next_key(struct bpf_map *map, void *key, void *next_key)
{
 struct bpf_htab *htab = container_of(map, struct bpf_htab, map);
 struct hlist_head *head;
 struct htab_elem *l, *next_l;
 u32 hash, key_size;
 int i;

 WARN_ON_ONCE(!rcu_read_lock_held());

 key_size = map->key_size;

 hash = htab_map_hash(key, key_size);

 head = select_bucket(htab, hash);


 l = lookup_elem_raw(head, hash, key, key_size);

 if (!l) {
  i = 0;
  goto find_first_elem;
 }


 next_l = hlist_entry_safe(rcu_dereference_raw(hlist_next_rcu(&l->hash_node)),
      struct htab_elem, hash_node);

 if (next_l) {

  memcpy(next_key, next_l->key, key_size);
  return 0;
 }


 i = hash & (htab->n_buckets - 1);
 i++;

find_first_elem:

 for (; i < htab->n_buckets; i++) {
  head = select_bucket(htab, i);


  next_l = hlist_entry_safe(rcu_dereference_raw(hlist_first_rcu(head)),
       struct htab_elem, hash_node);
  if (next_l) {

   memcpy(next_key, next_l->key, key_size);
   return 0;
  }
 }


 return -ENOENT;
}

static void htab_elem_free(struct bpf_htab *htab, struct htab_elem *l)
{
 if (htab->map.map_type == BPF_MAP_TYPE_PERCPU_HASH)
  free_percpu(htab_elem_get_ptr(l, htab->map.key_size));
 kfree(l);

}

static void htab_elem_free_rcu(struct rcu_head *head)
{
 struct htab_elem *l = container_of(head, struct htab_elem, rcu);
 struct bpf_htab *htab = l->htab;





 preempt_disable();
 __this_cpu_inc(bpf_prog_active);
 htab_elem_free(htab, l);
 __this_cpu_dec(bpf_prog_active);
 preempt_enable();
}

static void free_htab_elem(struct bpf_htab *htab, struct htab_elem *l)
{
 if (!(htab->map.map_flags & BPF_F_NO_PREALLOC)) {
  pcpu_freelist_push(&htab->freelist, &l->fnode);
 } else {
  atomic_dec(&htab->count);
  l->htab = htab;
  call_rcu(&l->rcu, htab_elem_free_rcu);
 }
}

static struct htab_elem *alloc_htab_elem(struct bpf_htab *htab, void *key,
      void *value, u32 key_size, u32 hash,
      bool percpu, bool onallcpus)
{
 u32 size = htab->map.value_size;
 bool prealloc = !(htab->map.map_flags & BPF_F_NO_PREALLOC);
 struct htab_elem *l_new;
 void __percpu *pptr;

 if (prealloc) {
  l_new = (struct htab_elem *)pcpu_freelist_pop(&htab->freelist);
  if (!l_new)
   return ERR_PTR(-E2BIG);
 } else {
  if (atomic_inc_return(&htab->count) > htab->map.max_entries) {
   atomic_dec(&htab->count);
   return ERR_PTR(-E2BIG);
  }
  l_new = kmalloc(htab->elem_size, GFP_ATOMIC | __GFP_NOWARN);
  if (!l_new)
   return ERR_PTR(-ENOMEM);
 }

 memcpy(l_new->key, key, key_size);
 if (percpu) {

  size = round_up(size, 8);

  if (prealloc) {
   pptr = htab_elem_get_ptr(l_new, key_size);
  } else {

   pptr = __alloc_percpu_gfp(size, 8,
        GFP_ATOMIC | __GFP_NOWARN);
   if (!pptr) {
    kfree(l_new);
    return ERR_PTR(-ENOMEM);
   }
  }

  if (!onallcpus) {

   memcpy(this_cpu_ptr(pptr), value, htab->map.value_size);
  } else {
   int off = 0, cpu;

   for_each_possible_cpu(cpu) {
    bpf_long_memcpy(per_cpu_ptr(pptr, cpu),
      value + off, size);
    off += size;
   }
  }
  if (!prealloc)
   htab_elem_set_ptr(l_new, key_size, pptr);
 } else {
  memcpy(l_new->key + round_up(key_size, 8), value, size);
 }

 l_new->hash = hash;
 return l_new;
}

static int check_flags(struct bpf_htab *htab, struct htab_elem *l_old,
         u64 map_flags)
{
 if (l_old && map_flags == BPF_NOEXIST)

  return -EEXIST;

 if (!l_old && map_flags == BPF_EXIST)

  return -ENOENT;

 return 0;
}


static int htab_map_update_elem(struct bpf_map *map, void *key, void *value,
    u64 map_flags)
{
 struct bpf_htab *htab = container_of(map, struct bpf_htab, map);
 struct htab_elem *l_new = NULL, *l_old;
 struct hlist_head *head;
 unsigned long flags;
 struct bucket *b;
 u32 key_size, hash;
 int ret;

 if (unlikely(map_flags > BPF_EXIST))

  return -EINVAL;

 WARN_ON_ONCE(!rcu_read_lock_held());

 key_size = map->key_size;

 hash = htab_map_hash(key, key_size);

 b = __select_bucket(htab, hash);
 head = &b->head;


 raw_spin_lock_irqsave(&b->lock, flags);

 l_old = lookup_elem_raw(head, hash, key, key_size);

 ret = check_flags(htab, l_old, map_flags);
 if (ret)
  goto err;

 l_new = alloc_htab_elem(htab, key, value, key_size, hash, false, false);
 if (IS_ERR(l_new)) {

  ret = PTR_ERR(l_new);
  goto err;
 }




 hlist_add_head_rcu(&l_new->hash_node, head);
 if (l_old) {
  hlist_del_rcu(&l_old->hash_node);
  free_htab_elem(htab, l_old);
 }
 ret = 0;
err:
 raw_spin_unlock_irqrestore(&b->lock, flags);
 return ret;
}

static int __htab_percpu_map_update_elem(struct bpf_map *map, void *key,
      void *value, u64 map_flags,
      bool onallcpus)
{
 struct bpf_htab *htab = container_of(map, struct bpf_htab, map);
 struct htab_elem *l_new = NULL, *l_old;
 struct hlist_head *head;
 unsigned long flags;
 struct bucket *b;
 u32 key_size, hash;
 int ret;

 if (unlikely(map_flags > BPF_EXIST))

  return -EINVAL;

 WARN_ON_ONCE(!rcu_read_lock_held());

 key_size = map->key_size;

 hash = htab_map_hash(key, key_size);

 b = __select_bucket(htab, hash);
 head = &b->head;


 raw_spin_lock_irqsave(&b->lock, flags);

 l_old = lookup_elem_raw(head, hash, key, key_size);

 ret = check_flags(htab, l_old, map_flags);
 if (ret)
  goto err;

 if (l_old) {
  void __percpu *pptr = htab_elem_get_ptr(l_old, key_size);
  u32 size = htab->map.value_size;


  if (!onallcpus) {
   memcpy(this_cpu_ptr(pptr), value, size);
  } else {
   int off = 0, cpu;

   size = round_up(size, 8);
   for_each_possible_cpu(cpu) {
    bpf_long_memcpy(per_cpu_ptr(pptr, cpu),
      value + off, size);
    off += size;
   }
  }
 } else {
  l_new = alloc_htab_elem(htab, key, value, key_size,
     hash, true, onallcpus);
  if (IS_ERR(l_new)) {
   ret = PTR_ERR(l_new);
   goto err;
  }
  hlist_add_head_rcu(&l_new->hash_node, head);
 }
 ret = 0;
err:
 raw_spin_unlock_irqrestore(&b->lock, flags);
 return ret;
}

static int htab_percpu_map_update_elem(struct bpf_map *map, void *key,
           void *value, u64 map_flags)
{
 return __htab_percpu_map_update_elem(map, key, value, map_flags, false);
}


static int htab_map_delete_elem(struct bpf_map *map, void *key)
{
 struct bpf_htab *htab = container_of(map, struct bpf_htab, map);
 struct hlist_head *head;
 struct bucket *b;
 struct htab_elem *l;
 unsigned long flags;
 u32 hash, key_size;
 int ret = -ENOENT;

 WARN_ON_ONCE(!rcu_read_lock_held());

 key_size = map->key_size;

 hash = htab_map_hash(key, key_size);
 b = __select_bucket(htab, hash);
 head = &b->head;

 raw_spin_lock_irqsave(&b->lock, flags);

 l = lookup_elem_raw(head, hash, key, key_size);

 if (l) {
  hlist_del_rcu(&l->hash_node);
  free_htab_elem(htab, l);
  ret = 0;
 }

 raw_spin_unlock_irqrestore(&b->lock, flags);
 return ret;
}

static void delete_all_elements(struct bpf_htab *htab)
{
 int i;

 for (i = 0; i < htab->n_buckets; i++) {
  struct hlist_head *head = select_bucket(htab, i);
  struct hlist_node *n;
  struct htab_elem *l;

  hlist_for_each_entry_safe(l, n, head, hash_node) {
   hlist_del_rcu(&l->hash_node);
   htab_elem_free(htab, l);
  }
 }
}

static void htab_map_free(struct bpf_map *map)
{
 struct bpf_htab *htab = container_of(map, struct bpf_htab, map);






 synchronize_rcu();




 rcu_barrier();
 if (htab->map.map_flags & BPF_F_NO_PREALLOC) {
  delete_all_elements(htab);
 } else {
  htab_free_elems(htab);
  pcpu_freelist_destroy(&htab->freelist);
 }
 kvfree(htab->buckets);
 kfree(htab);
}

static const struct bpf_map_ops htab_ops = {
 .map_alloc = htab_map_alloc,
 .map_free = htab_map_free,
 .map_get_next_key = htab_map_get_next_key,
 .map_lookup_elem = htab_map_lookup_elem,
 .map_update_elem = htab_map_update_elem,
 .map_delete_elem = htab_map_delete_elem,
};

static struct bpf_map_type_list htab_type __read_mostly = {
 .ops = &htab_ops,
 .type = BPF_MAP_TYPE_HASH,
};


static void *htab_percpu_map_lookup_elem(struct bpf_map *map, void *key)
{
 struct htab_elem *l = __htab_map_lookup_elem(map, key);

 if (l)
  return this_cpu_ptr(htab_elem_get_ptr(l, map->key_size));
 else
  return NULL;
}

int bpf_percpu_hash_copy(struct bpf_map *map, void *key, void *value)
{
 struct htab_elem *l;
 void __percpu *pptr;
 int ret = -ENOENT;
 int cpu, off = 0;
 u32 size;





 size = round_up(map->value_size, 8);
 rcu_read_lock();
 l = __htab_map_lookup_elem(map, key);
 if (!l)
  goto out;
 pptr = htab_elem_get_ptr(l, map->key_size);
 for_each_possible_cpu(cpu) {
  bpf_long_memcpy(value + off,
    per_cpu_ptr(pptr, cpu), size);
  off += size;
 }
 ret = 0;
out:
 rcu_read_unlock();
 return ret;
}

int bpf_percpu_hash_update(struct bpf_map *map, void *key, void *value,
      u64 map_flags)
{
 int ret;

 rcu_read_lock();
 ret = __htab_percpu_map_update_elem(map, key, value, map_flags, true);
 rcu_read_unlock();

 return ret;
}

static const struct bpf_map_ops htab_percpu_ops = {
 .map_alloc = htab_map_alloc,
 .map_free = htab_map_free,
 .map_get_next_key = htab_map_get_next_key,
 .map_lookup_elem = htab_percpu_map_lookup_elem,
 .map_update_elem = htab_percpu_map_update_elem,
 .map_delete_elem = htab_map_delete_elem,
};

static struct bpf_map_type_list htab_percpu_type __read_mostly = {
 .ops = &htab_percpu_ops,
 .type = BPF_MAP_TYPE_PERCPU_HASH,
};

static int __init register_htab_map(void)
{
 bpf_register_map_type(&htab_type);
 bpf_register_map_type(&htab_percpu_type);
 return 0;
}
late_initcall(register_htab_map);
static u64 bpf_map_lookup_elem(u64 r1, u64 r2, u64 r3, u64 r4, u64 r5)
{




 struct bpf_map *map = (struct bpf_map *) (unsigned long) r1;
 void *key = (void *) (unsigned long) r2;
 void *value;

 WARN_ON_ONCE(!rcu_read_lock_held());

 value = map->ops->map_lookup_elem(map, key);




 return (unsigned long) value;
}

const struct bpf_func_proto bpf_map_lookup_elem_proto = {
 .func = bpf_map_lookup_elem,
 .gpl_only = false,
 .ret_type = RET_PTR_TO_MAP_VALUE_OR_NULL,
 .arg1_type = ARG_CONST_MAP_PTR,
 .arg2_type = ARG_PTR_TO_MAP_KEY,
};

static u64 bpf_map_update_elem(u64 r1, u64 r2, u64 r3, u64 r4, u64 r5)
{
 struct bpf_map *map = (struct bpf_map *) (unsigned long) r1;
 void *key = (void *) (unsigned long) r2;
 void *value = (void *) (unsigned long) r3;

 WARN_ON_ONCE(!rcu_read_lock_held());

 return map->ops->map_update_elem(map, key, value, r4);
}

const struct bpf_func_proto bpf_map_update_elem_proto = {
 .func = bpf_map_update_elem,
 .gpl_only = false,
 .ret_type = RET_INTEGER,
 .arg1_type = ARG_CONST_MAP_PTR,
 .arg2_type = ARG_PTR_TO_MAP_KEY,
 .arg3_type = ARG_PTR_TO_MAP_VALUE,
 .arg4_type = ARG_ANYTHING,
};

static u64 bpf_map_delete_elem(u64 r1, u64 r2, u64 r3, u64 r4, u64 r5)
{
 struct bpf_map *map = (struct bpf_map *) (unsigned long) r1;
 void *key = (void *) (unsigned long) r2;

 WARN_ON_ONCE(!rcu_read_lock_held());

 return map->ops->map_delete_elem(map, key);
}

const struct bpf_func_proto bpf_map_delete_elem_proto = {
 .func = bpf_map_delete_elem,
 .gpl_only = false,
 .ret_type = RET_INTEGER,
 .arg1_type = ARG_CONST_MAP_PTR,
 .arg2_type = ARG_PTR_TO_MAP_KEY,
};

const struct bpf_func_proto bpf_get_prandom_u32_proto = {
 .func = bpf_user_rnd_u32,
 .gpl_only = false,
 .ret_type = RET_INTEGER,
};

static u64 bpf_get_smp_processor_id(u64 r1, u64 r2, u64 r3, u64 r4, u64 r5)
{
 return raw_smp_processor_id();
}

const struct bpf_func_proto bpf_get_smp_processor_id_proto = {
 .func = bpf_get_smp_processor_id,
 .gpl_only = false,
 .ret_type = RET_INTEGER,
};

static u64 bpf_ktime_get_ns(u64 r1, u64 r2, u64 r3, u64 r4, u64 r5)
{

 return ktime_get_mono_fast_ns();
}

const struct bpf_func_proto bpf_ktime_get_ns_proto = {
 .func = bpf_ktime_get_ns,
 .gpl_only = true,
 .ret_type = RET_INTEGER,
};

static u64 bpf_get_current_pid_tgid(u64 r1, u64 r2, u64 r3, u64 r4, u64 r5)
{
 struct task_struct *task = current;

 if (!task)
  return -EINVAL;

 return (u64) task->tgid << 32 | task->pid;
}

const struct bpf_func_proto bpf_get_current_pid_tgid_proto = {
 .func = bpf_get_current_pid_tgid,
 .gpl_only = false,
 .ret_type = RET_INTEGER,
};

static u64 bpf_get_current_uid_gid(u64 r1, u64 r2, u64 r3, u64 r4, u64 r5)
{
 struct task_struct *task = current;
 kuid_t uid;
 kgid_t gid;

 if (!task)
  return -EINVAL;

 current_uid_gid(&uid, &gid);
 return (u64) from_kgid(&init_user_ns, gid) << 32 |
  from_kuid(&init_user_ns, uid);
}

const struct bpf_func_proto bpf_get_current_uid_gid_proto = {
 .func = bpf_get_current_uid_gid,
 .gpl_only = false,
 .ret_type = RET_INTEGER,
};

static u64 bpf_get_current_comm(u64 r1, u64 size, u64 r3, u64 r4, u64 r5)
{
 struct task_struct *task = current;
 char *buf = (char *) (long) r1;

 if (unlikely(!task))
  goto err_clear;

 strncpy(buf, task->comm, size);





 buf[size - 1] = 0;
 return 0;
err_clear:
 memset(buf, 0, size);
 return -EINVAL;
}

const struct bpf_func_proto bpf_get_current_comm_proto = {
 .func = bpf_get_current_comm,
 .gpl_only = false,
 .ret_type = RET_INTEGER,
 .arg1_type = ARG_PTR_TO_RAW_STACK,
 .arg2_type = ARG_CONST_STACK_SIZE,
};



static int nocompress;
static int noresume;
static int nohibernate;
static int resume_wait;
static unsigned int resume_delay;
static char resume_file[256] = CONFIG_PM_STD_PARTITION;
dev_t swsusp_resume_device;
sector_t swsusp_resume_block;
__visible int in_suspend __nosavedata;

enum {
 HIBERNATION_INVALID,
 HIBERNATION_PLATFORM,
 HIBERNATION_SHUTDOWN,
 HIBERNATION_REBOOT,
 HIBERNATION_SUSPEND,

 __HIBERNATION_AFTER_LAST
};

static int hibernation_mode = HIBERNATION_SHUTDOWN;

bool freezer_test_done;

static const struct platform_hibernation_ops *hibernation_ops;

bool hibernation_available(void)
{
 return (nohibernate == 0);
}





void hibernation_set_ops(const struct platform_hibernation_ops *ops)
{
 if (ops && !(ops->begin && ops->end && ops->pre_snapshot
     && ops->prepare && ops->finish && ops->enter && ops->pre_restore
     && ops->restore_cleanup && ops->leave)) {
  WARN_ON(1);
  return;
 }
 lock_system_sleep();
 hibernation_ops = ops;
 if (ops)
  hibernation_mode = HIBERNATION_PLATFORM;
 else if (hibernation_mode == HIBERNATION_PLATFORM)
  hibernation_mode = HIBERNATION_SHUTDOWN;

 unlock_system_sleep();
}
EXPORT_SYMBOL_GPL(hibernation_set_ops);

static bool entering_platform_hibernation;

bool system_entering_hibernation(void)
{
 return entering_platform_hibernation;
}
EXPORT_SYMBOL(system_entering_hibernation);

static void hibernation_debug_sleep(void)
{
 printk(KERN_INFO "hibernation debug: Waiting for 5 seconds.\n");
 mdelay(5000);
}

static int hibernation_test(int level)
{
 if (pm_test_level == level) {
  hibernation_debug_sleep();
  return 1;
 }
 return 0;
}
static int hibernation_test(int level) { return 0; }





static int platform_begin(int platform_mode)
{
 return (platform_mode && hibernation_ops) ?
  hibernation_ops->begin() : 0;
}





static void platform_end(int platform_mode)
{
 if (platform_mode && hibernation_ops)
  hibernation_ops->end();
}
static int platform_pre_snapshot(int platform_mode)
{
 return (platform_mode && hibernation_ops) ?
  hibernation_ops->pre_snapshot() : 0;
}
static void platform_leave(int platform_mode)
{
 if (platform_mode && hibernation_ops)
  hibernation_ops->leave();
}
static void platform_finish(int platform_mode)
{
 if (platform_mode && hibernation_ops)
  hibernation_ops->finish();
}
static int platform_pre_restore(int platform_mode)
{
 return (platform_mode && hibernation_ops) ?
  hibernation_ops->pre_restore() : 0;
}
static void platform_restore_cleanup(int platform_mode)
{
 if (platform_mode && hibernation_ops)
  hibernation_ops->restore_cleanup();
}





static void platform_recover(int platform_mode)
{
 if (platform_mode && hibernation_ops && hibernation_ops->recover)
  hibernation_ops->recover();
}
void swsusp_show_speed(ktime_t start, ktime_t stop,
        unsigned nr_pages, char *msg)
{
 ktime_t diff;
 u64 elapsed_centisecs64;
 unsigned int centisecs;
 unsigned int k;
 unsigned int kps;

 diff = ktime_sub(stop, start);
 elapsed_centisecs64 = ktime_divns(diff, 10*NSEC_PER_MSEC);
 centisecs = elapsed_centisecs64;
 if (centisecs == 0)
  centisecs = 1;
 k = nr_pages * (PAGE_SIZE / 1024);
 kps = (k * 100) / centisecs;
 printk(KERN_INFO "PM: %s %u kbytes in %u.%02u seconds (%u.%02u MB/s)\n",
   msg, k,
   centisecs / 100, centisecs % 100,
   kps / 1000, (kps % 1000) / 10);
}
static int create_image(int platform_mode)
{
 int error;

 error = dpm_suspend_end(PMSG_FREEZE);
 if (error) {
  printk(KERN_ERR "PM: Some devices failed to power down, "
   "aborting hibernation\n");
  return error;
 }

 error = platform_pre_snapshot(platform_mode);
 if (error || hibernation_test(TEST_PLATFORM))
  goto Platform_finish;

 error = disable_nonboot_cpus();
 if (error || hibernation_test(TEST_CPUS))
  goto Enable_cpus;

 local_irq_disable();

 error = syscore_suspend();
 if (error) {
  printk(KERN_ERR "PM: Some system devices failed to power down, "
   "aborting hibernation\n");
  goto Enable_irqs;
 }

 if (hibernation_test(TEST_CORE) || pm_wakeup_pending())
  goto Power_up;

 in_suspend = 1;
 save_processor_state();
 trace_suspend_resume(TPS("machine_suspend"), PM_EVENT_HIBERNATE, true);
 error = swsusp_arch_suspend();
 trace_suspend_resume(TPS("machine_suspend"), PM_EVENT_HIBERNATE, false);
 if (error)
  printk(KERN_ERR "PM: Error %d creating hibernation image\n",
   error);

 restore_processor_state();
 if (!in_suspend)
  events_check_enabled = false;

 platform_leave(platform_mode);

 Power_up:
 syscore_resume();

 Enable_irqs:
 local_irq_enable();

 Enable_cpus:
 enable_nonboot_cpus();

 Platform_finish:
 platform_finish(platform_mode);

 dpm_resume_start(in_suspend ?
  (error ? PMSG_RECOVER : PMSG_THAW) : PMSG_RESTORE);

 return error;
}







int hibernation_snapshot(int platform_mode)
{
 pm_message_t msg;
 int error;

 pm_suspend_clear_flags();
 error = platform_begin(platform_mode);
 if (error)
  goto Close;


 error = hibernate_preallocate_memory();
 if (error)
  goto Close;

 error = freeze_kernel_threads();
 if (error)
  goto Cleanup;

 if (hibernation_test(TEST_FREEZER)) {





  freezer_test_done = true;
  goto Thaw;
 }

 error = dpm_prepare(PMSG_FREEZE);
 if (error) {
  dpm_complete(PMSG_RECOVER);
  goto Thaw;
 }

 suspend_console();
 pm_restrict_gfp_mask();

 error = dpm_suspend(PMSG_FREEZE);

 if (error || hibernation_test(TEST_DEVICES))
  platform_recover(platform_mode);
 else
  error = create_image(platform_mode);
 if (error || !in_suspend)
  swsusp_free();

 msg = in_suspend ? (error ? PMSG_RECOVER : PMSG_THAW) : PMSG_RESTORE;
 dpm_resume(msg);

 if (error || !in_suspend)
  pm_restore_gfp_mask();

 resume_console();
 dpm_complete(msg);

 Close:
 platform_end(platform_mode);
 return error;

 Thaw:
 thaw_kernel_threads();
 Cleanup:
 swsusp_free();
 goto Close;
}
static int resume_target_kernel(bool platform_mode)
{
 int error;

 error = dpm_suspend_end(PMSG_QUIESCE);
 if (error) {
  printk(KERN_ERR "PM: Some devices failed to power down, "
   "aborting resume\n");
  return error;
 }

 error = platform_pre_restore(platform_mode);
 if (error)
  goto Cleanup;

 error = disable_nonboot_cpus();
 if (error)
  goto Enable_cpus;

 local_irq_disable();

 error = syscore_suspend();
 if (error)
  goto Enable_irqs;

 save_processor_state();
 error = restore_highmem();
 if (!error) {
  error = swsusp_arch_resume();





  BUG_ON(!error);




  restore_highmem();
 }





 swsusp_free();
 restore_processor_state();
 touch_softlockup_watchdog();

 syscore_resume();

 Enable_irqs:
 local_irq_enable();

 Enable_cpus:
 enable_nonboot_cpus();

 Cleanup:
 platform_restore_cleanup(platform_mode);

 dpm_resume_start(PMSG_RECOVER);

 return error;
}
int hibernation_restore(int platform_mode)
{
 int error;

 pm_prepare_console();
 suspend_console();
 pm_restrict_gfp_mask();
 error = dpm_suspend_start(PMSG_QUIESCE);
 if (!error) {
  error = resume_target_kernel(platform_mode);





  BUG_ON(!error);
 }
 dpm_resume_end(PMSG_RECOVER);
 pm_restore_gfp_mask();
 resume_console();
 pm_restore_console();
 return error;
}




int hibernation_platform_enter(void)
{
 int error;

 if (!hibernation_ops)
  return -ENOSYS;






 error = hibernation_ops->begin();
 if (error)
  goto Close;

 entering_platform_hibernation = true;
 suspend_console();
 error = dpm_suspend_start(PMSG_HIBERNATE);
 if (error) {
  if (hibernation_ops->recover)
   hibernation_ops->recover();
  goto Resume_devices;
 }

 error = dpm_suspend_end(PMSG_HIBERNATE);
 if (error)
  goto Resume_devices;

 error = hibernation_ops->prepare();
 if (error)
  goto Platform_finish;

 error = disable_nonboot_cpus();
 if (error)
  goto Enable_cpus;

 local_irq_disable();
 syscore_suspend();
 if (pm_wakeup_pending()) {
  error = -EAGAIN;
  goto Power_up;
 }

 hibernation_ops->enter();

 while (1);

 Power_up:
 syscore_resume();
 local_irq_enable();

 Enable_cpus:
 enable_nonboot_cpus();

 Platform_finish:
 hibernation_ops->finish();

 dpm_resume_start(PMSG_RESTORE);

 Resume_devices:
 entering_platform_hibernation = false;
 dpm_resume_end(PMSG_RESTORE);
 resume_console();

 Close:
 hibernation_ops->end();

 return error;
}
static void power_down(void)
{
 int error;

 switch (hibernation_mode) {
 case HIBERNATION_REBOOT:
  kernel_restart(NULL);
  break;
 case HIBERNATION_PLATFORM:
  hibernation_platform_enter();
 case HIBERNATION_SHUTDOWN:
  if (pm_power_off)
   kernel_power_off();
  break;
 case HIBERNATION_SUSPEND:
  error = suspend_devices_and_enter(PM_SUSPEND_MEM);
  if (error) {
   if (hibernation_ops)
    hibernation_mode = HIBERNATION_PLATFORM;
   else
    hibernation_mode = HIBERNATION_SHUTDOWN;
   power_down();
  }



  error = swsusp_unmark();
  if (error)
   printk(KERN_ERR "PM: Swap will be unusable! "
                   "Try swapon -a.\n");
  return;
 }
 kernel_halt();




 printk(KERN_CRIT "PM: Please power down manually\n");
 while (1)
  cpu_relax();
}




int hibernate(void)
{
 int error;

 if (!hibernation_available()) {
  pr_debug("PM: Hibernation not available.\n");
  return -EPERM;
 }

 lock_system_sleep();

 if (!atomic_add_unless(&snapshot_device_available, -1, 0)) {
  error = -EBUSY;
  goto Unlock;
 }

 pm_prepare_console();
 error = pm_notifier_call_chain(PM_HIBERNATION_PREPARE);
 if (error)
  goto Exit;

 printk(KERN_INFO "PM: Syncing filesystems ... ");
 sys_sync();
 printk("done.\n");

 error = freeze_processes();
 if (error)
  goto Exit;

 lock_device_hotplug();

 error = create_basic_memory_bitmaps();
 if (error)
  goto Thaw;

 error = hibernation_snapshot(hibernation_mode == HIBERNATION_PLATFORM);
 if (error || freezer_test_done)
  goto Free_bitmaps;

 if (in_suspend) {
  unsigned int flags = 0;

  if (hibernation_mode == HIBERNATION_PLATFORM)
   flags |= SF_PLATFORM_MODE;
  if (nocompress)
   flags |= SF_NOCOMPRESS_MODE;
  else
          flags |= SF_CRC32_MODE;

  pr_debug("PM: writing image.\n");
  error = swsusp_write(flags);
  swsusp_free();
  if (!error)
   power_down();
  in_suspend = 0;
  pm_restore_gfp_mask();
 } else {
  pr_debug("PM: Image restored successfully.\n");
 }

 Free_bitmaps:
 free_basic_memory_bitmaps();
 Thaw:
 unlock_device_hotplug();
 thaw_processes();


 freezer_test_done = false;
 Exit:
 pm_notifier_call_chain(PM_POST_HIBERNATION);
 pm_restore_console();
 atomic_inc(&snapshot_device_available);
 Unlock:
 unlock_system_sleep();
 return error;
}
static int software_resume(void)
{
 int error;
 unsigned int flags;




 if (noresume || !hibernation_available())
  return 0;
 mutex_lock_nested(&pm_mutex, SINGLE_DEPTH_NESTING);

 if (swsusp_resume_device)
  goto Check_image;

 if (!strlen(resume_file)) {
  error = -ENOENT;
  goto Unlock;
 }

 pr_debug("PM: Checking hibernation image partition %s\n", resume_file);

 if (resume_delay) {
  printk(KERN_INFO "Waiting %dsec before reading resume device...\n",
   resume_delay);
  ssleep(resume_delay);
 }


 swsusp_resume_device = name_to_dev_t(resume_file);





 if (isdigit(resume_file[0]) && resume_wait) {
  int partno;
  while (!get_gendisk(swsusp_resume_device, &partno))
   msleep(10);
 }

 if (!swsusp_resume_device) {




  wait_for_device_probe();

  if (resume_wait) {
   while ((swsusp_resume_device = name_to_dev_t(resume_file)) == 0)
    msleep(10);
   async_synchronize_full();
  }

  swsusp_resume_device = name_to_dev_t(resume_file);
  if (!swsusp_resume_device) {
   error = -ENODEV;
   goto Unlock;
  }
 }

 Check_image:
 pr_debug("PM: Hibernation image partition %d:%d present\n",
  MAJOR(swsusp_resume_device), MINOR(swsusp_resume_device));

 pr_debug("PM: Looking for hibernation image.\n");
 error = swsusp_check();
 if (error)
  goto Unlock;


 if (!atomic_add_unless(&snapshot_device_available, -1, 0)) {
  error = -EBUSY;
  swsusp_close(FMODE_READ);
  goto Unlock;
 }

 pm_prepare_console();
 error = pm_notifier_call_chain(PM_RESTORE_PREPARE);
 if (error)
  goto Close_Finish;

 pr_debug("PM: Preparing processes for restore.\n");
 error = freeze_processes();
 if (error)
  goto Close_Finish;

 pr_debug("PM: Loading hibernation image.\n");

 lock_device_hotplug();
 error = create_basic_memory_bitmaps();
 if (error)
  goto Thaw;

 error = swsusp_read(&flags);
 swsusp_close(FMODE_READ);
 if (!error)
  hibernation_restore(flags & SF_PLATFORM_MODE);

 printk(KERN_ERR "PM: Failed to load hibernation image, recovering.\n");
 swsusp_free();
 free_basic_memory_bitmaps();
 Thaw:
 unlock_device_hotplug();
 thaw_processes();
 Finish:
 pm_notifier_call_chain(PM_POST_RESTORE);
 pm_restore_console();
 atomic_inc(&snapshot_device_available);

 Unlock:
 mutex_unlock(&pm_mutex);
 pr_debug("PM: Hibernation image not present or could not be loaded.\n");
 return error;
 Close_Finish:
 swsusp_close(FMODE_READ);
 goto Finish;
}

late_initcall_sync(software_resume);


static const char * const hibernation_modes[] = {
 [HIBERNATION_PLATFORM] = "platform",
 [HIBERNATION_SHUTDOWN] = "shutdown",
 [HIBERNATION_REBOOT] = "reboot",
 [HIBERNATION_SUSPEND] = "suspend",
};
static ssize_t disk_show(struct kobject *kobj, struct kobj_attribute *attr,
    char *buf)
{
 int i;
 char *start = buf;

 if (!hibernation_available())
  return sprintf(buf, "[disabled]\n");

 for (i = HIBERNATION_FIRST; i <= HIBERNATION_MAX; i++) {
  if (!hibernation_modes[i])
   continue;
  switch (i) {
  case HIBERNATION_SHUTDOWN:
  case HIBERNATION_REBOOT:
  case HIBERNATION_SUSPEND:
   break;
  case HIBERNATION_PLATFORM:
   if (hibernation_ops)
    break;

   continue;
  }
  if (i == hibernation_mode)
   buf += sprintf(buf, "[%s] ", hibernation_modes[i]);
  else
   buf += sprintf(buf, "%s ", hibernation_modes[i]);
 }
 buf += sprintf(buf, "\n");
 return buf-start;
}

static ssize_t disk_store(struct kobject *kobj, struct kobj_attribute *attr,
     const char *buf, size_t n)
{
 int error = 0;
 int i;
 int len;
 char *p;
 int mode = HIBERNATION_INVALID;

 if (!hibernation_available())
  return -EPERM;

 p = memchr(buf, '\n', n);
 len = p ? p - buf : n;

 lock_system_sleep();
 for (i = HIBERNATION_FIRST; i <= HIBERNATION_MAX; i++) {
  if (len == strlen(hibernation_modes[i])
      && !strncmp(buf, hibernation_modes[i], len)) {
   mode = i;
   break;
  }
 }
 if (mode != HIBERNATION_INVALID) {
  switch (mode) {
  case HIBERNATION_SHUTDOWN:
  case HIBERNATION_REBOOT:
  case HIBERNATION_SUSPEND:
   hibernation_mode = mode;
   break;
  case HIBERNATION_PLATFORM:
   if (hibernation_ops)
    hibernation_mode = mode;
   else
    error = -EINVAL;
  }
 } else
  error = -EINVAL;

 if (!error)
  pr_debug("PM: Hibernation mode set to '%s'\n",
    hibernation_modes[mode]);
 unlock_system_sleep();
 return error ? error : n;
}

power_attr(disk);

static ssize_t resume_show(struct kobject *kobj, struct kobj_attribute *attr,
      char *buf)
{
 return sprintf(buf,"%d:%d\n", MAJOR(swsusp_resume_device),
         MINOR(swsusp_resume_device));
}

static ssize_t resume_store(struct kobject *kobj, struct kobj_attribute *attr,
       const char *buf, size_t n)
{
 dev_t res;
 int len = n;
 char *name;

 if (len && buf[len-1] == '\n')
  len--;
 name = kstrndup(buf, len, GFP_KERNEL);
 if (!name)
  return -ENOMEM;

 res = name_to_dev_t(name);
 kfree(name);
 if (!res)
  return -EINVAL;

 lock_system_sleep();
 swsusp_resume_device = res;
 unlock_system_sleep();
 printk(KERN_INFO "PM: Starting manual resume from disk\n");
 noresume = 0;
 software_resume();
 return n;
}

power_attr(resume);

static ssize_t image_size_show(struct kobject *kobj, struct kobj_attribute *attr,
          char *buf)
{
 return sprintf(buf, "%lu\n", image_size);
}

static ssize_t image_size_store(struct kobject *kobj, struct kobj_attribute *attr,
    const char *buf, size_t n)
{
 unsigned long size;

 if (sscanf(buf, "%lu", &size) == 1) {
  image_size = size;
  return n;
 }

 return -EINVAL;
}

power_attr(image_size);

static ssize_t reserved_size_show(struct kobject *kobj,
      struct kobj_attribute *attr, char *buf)
{
 return sprintf(buf, "%lu\n", reserved_size);
}

static ssize_t reserved_size_store(struct kobject *kobj,
       struct kobj_attribute *attr,
       const char *buf, size_t n)
{
 unsigned long size;

 if (sscanf(buf, "%lu", &size) == 1) {
  reserved_size = size;
  return n;
 }

 return -EINVAL;
}

power_attr(reserved_size);

static struct attribute * g[] = {
 &disk_attr.attr,
 &resume_attr.attr,
 &image_size_attr.attr,
 &reserved_size_attr.attr,
 NULL,
};


static struct attribute_group attr_group = {
 .attrs = g,
};


static int __init pm_disk_init(void)
{
 return sysfs_create_group(power_kobj, &attr_group);
}

core_initcall(pm_disk_init);


static int __init resume_setup(char *str)
{
 if (noresume)
  return 1;

 strncpy( resume_file, str, 255 );
 return 1;
}

static int __init resume_offset_setup(char *str)
{
 unsigned long long offset;

 if (noresume)
  return 1;

 if (sscanf(str, "%llu", &offset) == 1)
  swsusp_resume_block = offset;

 return 1;
}

static int __init hibernate_setup(char *str)
{
 if (!strncmp(str, "noresume", 8))
  noresume = 1;
 else if (!strncmp(str, "nocompress", 10))
  nocompress = 1;
 else if (!strncmp(str, "no", 2)) {
  noresume = 1;
  nohibernate = 1;
 }
 return 1;
}

static int __init noresume_setup(char *str)
{
 noresume = 1;
 return 1;
}

static int __init resumewait_setup(char *str)
{
 resume_wait = 1;
 return 1;
}

static int __init resumedelay_setup(char *str)
{
 int rc = kstrtouint(str, 0, &resume_delay);

 if (rc)
  return rc;
 return 1;
}

static int __init nohibernate_setup(char *str)
{
 noresume = 1;
 nohibernate = 1;
 return 1;
}

static int __init kaslr_nohibernate_setup(char *str)
{
 return nohibernate_setup(str);
}

static int __init page_poison_nohibernate_setup(char *str)
{





 if (!strcmp(str, "on")) {
  pr_info("Disabling hibernation due to page poisoning\n");
  return nohibernate_setup(str);
 }
 return 1;
}

__setup("noresume", noresume_setup);
__setup("resume_offset=", resume_offset_setup);
__setup("resume=", resume_setup);
__setup("hibernate=", hibernate_setup);
__setup("resumewait", resumewait_setup);
__setup("resumedelay=", resumedelay_setup);
__setup("nohibernate", nohibernate_setup);
__setup("kaslr", kaslr_nohibernate_setup);
__setup("page_poison=", page_poison_nohibernate_setup);



DEFINE_PER_CPU(struct hrtimer_cpu_base, hrtimer_bases) =
{
 .lock = __RAW_SPIN_LOCK_UNLOCKED(hrtimer_bases.lock),
 .seq = SEQCNT_ZERO(hrtimer_bases.seq),
 .clock_base =
 {
  {
   .index = HRTIMER_BASE_MONOTONIC,
   .clockid = CLOCK_MONOTONIC,
   .get_time = &ktime_get,
  },
  {
   .index = HRTIMER_BASE_REALTIME,
   .clockid = CLOCK_REALTIME,
   .get_time = &ktime_get_real,
  },
  {
   .index = HRTIMER_BASE_BOOTTIME,
   .clockid = CLOCK_BOOTTIME,
   .get_time = &ktime_get_boottime,
  },
  {
   .index = HRTIMER_BASE_TAI,
   .clockid = CLOCK_TAI,
   .get_time = &ktime_get_clocktai,
  },
 }
};

static const int hrtimer_clock_to_base_table[MAX_CLOCKS] = {
 [CLOCK_REALTIME] = HRTIMER_BASE_REALTIME,
 [CLOCK_MONOTONIC] = HRTIMER_BASE_MONOTONIC,
 [CLOCK_BOOTTIME] = HRTIMER_BASE_BOOTTIME,
 [CLOCK_TAI] = HRTIMER_BASE_TAI,
};

static inline int hrtimer_clockid_to_base(clockid_t clock_id)
{
 return hrtimer_clock_to_base_table[clock_id];
}











static struct hrtimer_cpu_base migration_cpu_base = {
 .seq = SEQCNT_ZERO(migration_cpu_base),
 .clock_base = { { .cpu_base = &migration_cpu_base, }, },
};

static
struct hrtimer_clock_base *lock_hrtimer_base(const struct hrtimer *timer,
          unsigned long *flags)
{
 struct hrtimer_clock_base *base;

 for (;;) {
  base = timer->base;
  if (likely(base != &migration_base)) {
   raw_spin_lock_irqsave(&base->cpu_base->lock, *flags);
   if (likely(base == timer->base))
    return base;

   raw_spin_unlock_irqrestore(&base->cpu_base->lock, *flags);
  }
  cpu_relax();
 }
}
static int
hrtimer_check_target(struct hrtimer *timer, struct hrtimer_clock_base *new_base)
{
 ktime_t expires;

 if (!new_base->cpu_base->hres_active)
  return 0;

 expires = ktime_sub(hrtimer_get_expires(timer), new_base->offset);
 return expires.tv64 <= new_base->cpu_base->expires_next.tv64;
 return 0;
}

static inline
struct hrtimer_cpu_base *get_target_base(struct hrtimer_cpu_base *base,
      int pinned)
{
 if (pinned || !base->migration_enabled)
  return base;
 return &per_cpu(hrtimer_bases, get_nohz_timer_target());
}
static inline
struct hrtimer_cpu_base *get_target_base(struct hrtimer_cpu_base *base,
      int pinned)
{
 return base;
}
static inline struct hrtimer_clock_base *
switch_hrtimer_base(struct hrtimer *timer, struct hrtimer_clock_base *base,
      int pinned)
{
 struct hrtimer_cpu_base *new_cpu_base, *this_cpu_base;
 struct hrtimer_clock_base *new_base;
 int basenum = base->index;

 this_cpu_base = this_cpu_ptr(&hrtimer_bases);
 new_cpu_base = get_target_base(this_cpu_base, pinned);
again:
 new_base = &new_cpu_base->clock_base[basenum];

 if (base != new_base) {
  if (unlikely(hrtimer_callback_running(timer)))
   return base;


  timer->base = &migration_base;
  raw_spin_unlock(&base->cpu_base->lock);
  raw_spin_lock(&new_base->cpu_base->lock);

  if (new_cpu_base != this_cpu_base &&
      hrtimer_check_target(timer, new_base)) {
   raw_spin_unlock(&new_base->cpu_base->lock);
   raw_spin_lock(&base->cpu_base->lock);
   new_cpu_base = this_cpu_base;
   timer->base = base;
   goto again;
  }
  timer->base = new_base;
 } else {
  if (new_cpu_base != this_cpu_base &&
      hrtimer_check_target(timer, new_base)) {
   new_cpu_base = this_cpu_base;
   goto again;
  }
 }
 return new_base;
}


static inline struct hrtimer_clock_base *
lock_hrtimer_base(const struct hrtimer *timer, unsigned long *flags)
{
 struct hrtimer_clock_base *base = timer->base;

 raw_spin_lock_irqsave(&base->cpu_base->lock, *flags);

 return base;
}










s64 __ktime_divns(const ktime_t kt, s64 div)
{
 int sft = 0;
 s64 dclc;
 u64 tmp;

 dclc = ktime_to_ns(kt);
 tmp = dclc < 0 ? -dclc : dclc;


 while (div >> 32) {
  sft++;
  div >>= 1;
 }
 tmp >>= sft;
 do_div(tmp, (unsigned long) div);
 return dclc < 0 ? -tmp : tmp;
}
EXPORT_SYMBOL_GPL(__ktime_divns);




ktime_t ktime_add_safe(const ktime_t lhs, const ktime_t rhs)
{
 ktime_t res = ktime_add(lhs, rhs);





 if (res.tv64 < 0 || res.tv64 < lhs.tv64 || res.tv64 < rhs.tv64)
  res = ktime_set(KTIME_SEC_MAX, 0);

 return res;
}

EXPORT_SYMBOL_GPL(ktime_add_safe);


static struct debug_obj_descr hrtimer_debug_descr;

static void *hrtimer_debug_hint(void *addr)
{
 return ((struct hrtimer *) addr)->function;
}





static bool hrtimer_fixup_init(void *addr, enum debug_obj_state state)
{
 struct hrtimer *timer = addr;

 switch (state) {
 case ODEBUG_STATE_ACTIVE:
  hrtimer_cancel(timer);
  debug_object_init(timer, &hrtimer_debug_descr);
  return true;
 default:
  return false;
 }
}






static bool hrtimer_fixup_activate(void *addr, enum debug_obj_state state)
{
 switch (state) {
 case ODEBUG_STATE_ACTIVE:
  WARN_ON(1);

 default:
  return false;
 }
}





static bool hrtimer_fixup_free(void *addr, enum debug_obj_state state)
{
 struct hrtimer *timer = addr;

 switch (state) {
 case ODEBUG_STATE_ACTIVE:
  hrtimer_cancel(timer);
  debug_object_free(timer, &hrtimer_debug_descr);
  return true;
 default:
  return false;
 }
}

static struct debug_obj_descr hrtimer_debug_descr = {
 .name = "hrtimer",
 .debug_hint = hrtimer_debug_hint,
 .fixup_init = hrtimer_fixup_init,
 .fixup_activate = hrtimer_fixup_activate,
 .fixup_free = hrtimer_fixup_free,
};

static inline void debug_hrtimer_init(struct hrtimer *timer)
{
 debug_object_init(timer, &hrtimer_debug_descr);
}

static inline void debug_hrtimer_activate(struct hrtimer *timer)
{
 debug_object_activate(timer, &hrtimer_debug_descr);
}

static inline void debug_hrtimer_deactivate(struct hrtimer *timer)
{
 debug_object_deactivate(timer, &hrtimer_debug_descr);
}

static inline void debug_hrtimer_free(struct hrtimer *timer)
{
 debug_object_free(timer, &hrtimer_debug_descr);
}

static void __hrtimer_init(struct hrtimer *timer, clockid_t clock_id,
      enum hrtimer_mode mode);

void hrtimer_init_on_stack(struct hrtimer *timer, clockid_t clock_id,
      enum hrtimer_mode mode)
{
 debug_object_init_on_stack(timer, &hrtimer_debug_descr);
 __hrtimer_init(timer, clock_id, mode);
}
EXPORT_SYMBOL_GPL(hrtimer_init_on_stack);

void destroy_hrtimer_on_stack(struct hrtimer *timer)
{
 debug_object_free(timer, &hrtimer_debug_descr);
}
EXPORT_SYMBOL_GPL(destroy_hrtimer_on_stack);

static inline void debug_hrtimer_init(struct hrtimer *timer) { }
static inline void debug_hrtimer_activate(struct hrtimer *timer) { }
static inline void debug_hrtimer_deactivate(struct hrtimer *timer) { }

static inline void
debug_init(struct hrtimer *timer, clockid_t clockid,
    enum hrtimer_mode mode)
{
 debug_hrtimer_init(timer);
 trace_hrtimer_init(timer, clockid, mode);
}

static inline void debug_activate(struct hrtimer *timer)
{
 debug_hrtimer_activate(timer);
 trace_hrtimer_start(timer);
}

static inline void debug_deactivate(struct hrtimer *timer)
{
 debug_hrtimer_deactivate(timer);
 trace_hrtimer_cancel(timer);
}

static inline void hrtimer_update_next_timer(struct hrtimer_cpu_base *cpu_base,
          struct hrtimer *timer)
{
 cpu_base->next_timer = timer;
}

static ktime_t __hrtimer_get_next_event(struct hrtimer_cpu_base *cpu_base)
{
 struct hrtimer_clock_base *base = cpu_base->clock_base;
 ktime_t expires, expires_next = { .tv64 = KTIME_MAX };
 unsigned int active = cpu_base->active_bases;

 hrtimer_update_next_timer(cpu_base, NULL);
 for (; active; base++, active >>= 1) {
  struct timerqueue_node *next;
  struct hrtimer *timer;

  if (!(active & 0x01))
   continue;

  next = timerqueue_getnext(&base->active);
  timer = container_of(next, struct hrtimer, node);
  expires = ktime_sub(hrtimer_get_expires(timer), base->offset);
  if (expires.tv64 < expires_next.tv64) {
   expires_next = expires;
   hrtimer_update_next_timer(cpu_base, timer);
  }
 }





 if (expires_next.tv64 < 0)
  expires_next.tv64 = 0;
 return expires_next;
}

static inline ktime_t hrtimer_update_base(struct hrtimer_cpu_base *base)
{
 ktime_t *offs_real = &base->clock_base[HRTIMER_BASE_REALTIME].offset;
 ktime_t *offs_boot = &base->clock_base[HRTIMER_BASE_BOOTTIME].offset;
 ktime_t *offs_tai = &base->clock_base[HRTIMER_BASE_TAI].offset;

 return ktime_get_update_offsets_now(&base->clock_was_set_seq,
         offs_real, offs_boot, offs_tai);
}






static bool hrtimer_hres_enabled __read_mostly = true;
unsigned int hrtimer_resolution __read_mostly = LOW_RES_NSEC;
EXPORT_SYMBOL_GPL(hrtimer_resolution);




static int __init setup_hrtimer_hres(char *str)
{
 return (kstrtobool(str, &hrtimer_hres_enabled) == 0);
}

__setup("highres=", setup_hrtimer_hres);




static inline int hrtimer_is_hres_enabled(void)
{
 return hrtimer_hres_enabled;
}




static inline int __hrtimer_hres_active(struct hrtimer_cpu_base *cpu_base)
{
 return cpu_base->hres_active;
}

static inline int hrtimer_hres_active(void)
{
 return __hrtimer_hres_active(this_cpu_ptr(&hrtimer_bases));
}






static void
hrtimer_force_reprogram(struct hrtimer_cpu_base *cpu_base, int skip_equal)
{
 ktime_t expires_next;

 if (!cpu_base->hres_active)
  return;

 expires_next = __hrtimer_get_next_event(cpu_base);

 if (skip_equal && expires_next.tv64 == cpu_base->expires_next.tv64)
  return;

 cpu_base->expires_next.tv64 = expires_next.tv64;
 if (cpu_base->hang_detected)
  return;

 tick_program_event(cpu_base->expires_next, 1);
}
static void hrtimer_reprogram(struct hrtimer *timer,
         struct hrtimer_clock_base *base)
{
 struct hrtimer_cpu_base *cpu_base = this_cpu_ptr(&hrtimer_bases);
 ktime_t expires = ktime_sub(hrtimer_get_expires(timer), base->offset);

 WARN_ON_ONCE(hrtimer_get_expires_tv64(timer) < 0);





 if (base->cpu_base != cpu_base)
  return;
 if (cpu_base->in_hrtirq)
  return;





 if (expires.tv64 < 0)
  expires.tv64 = 0;

 if (expires.tv64 >= cpu_base->expires_next.tv64)
  return;


 cpu_base->next_timer = timer;







 if (cpu_base->hang_detected)
  return;





 cpu_base->expires_next = expires;
 tick_program_event(expires, 1);
}




static inline void hrtimer_init_hres(struct hrtimer_cpu_base *base)
{
 base->expires_next.tv64 = KTIME_MAX;
 base->hres_active = 0;
}






static void retrigger_next_event(void *arg)
{
 struct hrtimer_cpu_base *base = this_cpu_ptr(&hrtimer_bases);

 if (!base->hres_active)
  return;

 raw_spin_lock(&base->lock);
 hrtimer_update_base(base);
 hrtimer_force_reprogram(base, 0);
 raw_spin_unlock(&base->lock);
}




static void hrtimer_switch_to_hres(void)
{
 struct hrtimer_cpu_base *base = this_cpu_ptr(&hrtimer_bases);

 if (tick_init_highres()) {
  printk(KERN_WARNING "Could not switch to high resolution "
        "mode on CPU %d\n", base->cpu);
  return;
 }
 base->hres_active = 1;
 hrtimer_resolution = HIGH_RES_NSEC;

 tick_setup_sched_timer();

 retrigger_next_event(NULL);
}

static void clock_was_set_work(struct work_struct *work)
{
 clock_was_set();
}

static DECLARE_WORK(hrtimer_work, clock_was_set_work);





void clock_was_set_delayed(void)
{
 schedule_work(&hrtimer_work);
}


static inline int __hrtimer_hres_active(struct hrtimer_cpu_base *b) { return 0; }
static inline int hrtimer_hres_active(void) { return 0; }
static inline int hrtimer_is_hres_enabled(void) { return 0; }
static inline void hrtimer_switch_to_hres(void) { }
static inline void
hrtimer_force_reprogram(struct hrtimer_cpu_base *base, int skip_equal) { }
static inline int hrtimer_reprogram(struct hrtimer *timer,
        struct hrtimer_clock_base *base)
{
 return 0;
}
static inline void hrtimer_init_hres(struct hrtimer_cpu_base *base) { }
static inline void retrigger_next_event(void *arg) { }

void clock_was_set(void)
{

 on_each_cpu(retrigger_next_event, NULL, 1);
 timerfd_clock_was_set();
}







void hrtimers_resume(void)
{
 WARN_ONCE(!irqs_disabled(),
    KERN_INFO "hrtimers_resume() called with IRQs enabled!");


 retrigger_next_event(NULL);

 clock_was_set_delayed();
}

static inline void timer_stats_hrtimer_set_start_info(struct hrtimer *timer)
{
 if (timer->start_site)
  return;
 timer->start_site = __builtin_return_address(0);
 memcpy(timer->start_comm, current->comm, TASK_COMM_LEN);
 timer->start_pid = current->pid;
}

static inline void timer_stats_hrtimer_clear_start_info(struct hrtimer *timer)
{
 timer->start_site = NULL;
}

static inline void timer_stats_account_hrtimer(struct hrtimer *timer)
{
 if (likely(!timer_stats_active))
  return;
 timer_stats_update_stats(timer, timer->start_pid, timer->start_site,
     timer->function, timer->start_comm, 0);
}




static inline
void unlock_hrtimer_base(const struct hrtimer *timer, unsigned long *flags)
{
 raw_spin_unlock_irqrestore(&timer->base->cpu_base->lock, *flags);
}
u64 hrtimer_forward(struct hrtimer *timer, ktime_t now, ktime_t interval)
{
 u64 orun = 1;
 ktime_t delta;

 delta = ktime_sub(now, hrtimer_get_expires(timer));

 if (delta.tv64 < 0)
  return 0;

 if (WARN_ON(timer->state & HRTIMER_STATE_ENQUEUED))
  return 0;

 if (interval.tv64 < hrtimer_resolution)
  interval.tv64 = hrtimer_resolution;

 if (unlikely(delta.tv64 >= interval.tv64)) {
  s64 incr = ktime_to_ns(interval);

  orun = ktime_divns(delta, incr);
  hrtimer_add_expires_ns(timer, incr * orun);
  if (hrtimer_get_expires_tv64(timer) > now.tv64)
   return orun;




  orun++;
 }
 hrtimer_add_expires(timer, interval);

 return orun;
}
EXPORT_SYMBOL_GPL(hrtimer_forward);
static int enqueue_hrtimer(struct hrtimer *timer,
      struct hrtimer_clock_base *base)
{
 debug_activate(timer);

 base->cpu_base->active_bases |= 1 << base->index;

 timer->state = HRTIMER_STATE_ENQUEUED;

 return timerqueue_add(&base->active, &timer->node);
}
static void __remove_hrtimer(struct hrtimer *timer,
        struct hrtimer_clock_base *base,
        u8 newstate, int reprogram)
{
 struct hrtimer_cpu_base *cpu_base = base->cpu_base;
 u8 state = timer->state;

 timer->state = newstate;
 if (!(state & HRTIMER_STATE_ENQUEUED))
  return;

 if (!timerqueue_del(&base->active, &timer->node))
  cpu_base->active_bases &= ~(1 << base->index);

 if (reprogram && timer == cpu_base->next_timer)
  hrtimer_force_reprogram(cpu_base, 1);
}




static inline int
remove_hrtimer(struct hrtimer *timer, struct hrtimer_clock_base *base, bool restart)
{
 if (hrtimer_is_queued(timer)) {
  u8 state = timer->state;
  int reprogram;
  debug_deactivate(timer);
  timer_stats_hrtimer_clear_start_info(timer);
  reprogram = base->cpu_base == this_cpu_ptr(&hrtimer_bases);

  if (!restart)
   state = HRTIMER_STATE_INACTIVE;

  __remove_hrtimer(timer, base, state, reprogram);
  return 1;
 }
 return 0;
}

static inline ktime_t hrtimer_update_lowres(struct hrtimer *timer, ktime_t tim,
         const enum hrtimer_mode mode)
{





 timer->is_rel = mode & HRTIMER_MODE_REL;
 if (timer->is_rel)
  tim = ktime_add_safe(tim, ktime_set(0, hrtimer_resolution));
 return tim;
}
void hrtimer_start_range_ns(struct hrtimer *timer, ktime_t tim,
       u64 delta_ns, const enum hrtimer_mode mode)
{
 struct hrtimer_clock_base *base, *new_base;
 unsigned long flags;
 int leftmost;

 base = lock_hrtimer_base(timer, &flags);


 remove_hrtimer(timer, base, true);

 if (mode & HRTIMER_MODE_REL)
  tim = ktime_add_safe(tim, base->get_time());

 tim = hrtimer_update_lowres(timer, tim, mode);

 hrtimer_set_expires_range_ns(timer, tim, delta_ns);


 new_base = switch_hrtimer_base(timer, base, mode & HRTIMER_MODE_PINNED);

 timer_stats_hrtimer_set_start_info(timer);

 leftmost = enqueue_hrtimer(timer, new_base);
 if (!leftmost)
  goto unlock;

 if (!hrtimer_is_hres_active(timer)) {




  if (new_base->cpu_base->nohz_active)
   wake_up_nohz_cpu(new_base->cpu_base->cpu);
 } else {
  hrtimer_reprogram(timer, new_base);
 }
unlock:
 unlock_hrtimer_base(timer, &flags);
}
EXPORT_SYMBOL_GPL(hrtimer_start_range_ns);
int hrtimer_try_to_cancel(struct hrtimer *timer)
{
 struct hrtimer_clock_base *base;
 unsigned long flags;
 int ret = -1;







 if (!hrtimer_active(timer))
  return 0;

 base = lock_hrtimer_base(timer, &flags);

 if (!hrtimer_callback_running(timer))
  ret = remove_hrtimer(timer, base, false);

 unlock_hrtimer_base(timer, &flags);

 return ret;

}
EXPORT_SYMBOL_GPL(hrtimer_try_to_cancel);
int hrtimer_cancel(struct hrtimer *timer)
{
 for (;;) {
  int ret = hrtimer_try_to_cancel(timer);

  if (ret >= 0)
   return ret;
  cpu_relax();
 }
}
EXPORT_SYMBOL_GPL(hrtimer_cancel);






ktime_t __hrtimer_get_remaining(const struct hrtimer *timer, bool adjust)
{
 unsigned long flags;
 ktime_t rem;

 lock_hrtimer_base(timer, &flags);
 if (IS_ENABLED(CONFIG_TIME_LOW_RES) && adjust)
  rem = hrtimer_expires_remaining_adjusted(timer);
 else
  rem = hrtimer_expires_remaining(timer);
 unlock_hrtimer_base(timer, &flags);

 return rem;
}
EXPORT_SYMBOL_GPL(__hrtimer_get_remaining);






u64 hrtimer_get_next_event(void)
{
 struct hrtimer_cpu_base *cpu_base = this_cpu_ptr(&hrtimer_bases);
 u64 expires = KTIME_MAX;
 unsigned long flags;

 raw_spin_lock_irqsave(&cpu_base->lock, flags);

 if (!__hrtimer_hres_active(cpu_base))
  expires = __hrtimer_get_next_event(cpu_base).tv64;

 raw_spin_unlock_irqrestore(&cpu_base->lock, flags);

 return expires;
}

static void __hrtimer_init(struct hrtimer *timer, clockid_t clock_id,
      enum hrtimer_mode mode)
{
 struct hrtimer_cpu_base *cpu_base;
 int base;

 memset(timer, 0, sizeof(struct hrtimer));

 cpu_base = raw_cpu_ptr(&hrtimer_bases);

 if (clock_id == CLOCK_REALTIME && mode != HRTIMER_MODE_ABS)
  clock_id = CLOCK_MONOTONIC;

 base = hrtimer_clockid_to_base(clock_id);
 timer->base = &cpu_base->clock_base[base];
 timerqueue_init(&timer->node);

 timer->start_site = NULL;
 timer->start_pid = -1;
 memset(timer->start_comm, 0, TASK_COMM_LEN);
}







void hrtimer_init(struct hrtimer *timer, clockid_t clock_id,
    enum hrtimer_mode mode)
{
 debug_init(timer, clock_id, mode);
 __hrtimer_init(timer, clock_id, mode);
}
EXPORT_SYMBOL_GPL(hrtimer_init);
bool hrtimer_active(const struct hrtimer *timer)
{
 struct hrtimer_cpu_base *cpu_base;
 unsigned int seq;

 do {
  cpu_base = READ_ONCE(timer->base->cpu_base);
  seq = raw_read_seqcount_begin(&cpu_base->seq);

  if (timer->state != HRTIMER_STATE_INACTIVE ||
      cpu_base->running == timer)
   return true;

 } while (read_seqcount_retry(&cpu_base->seq, seq) ||
   cpu_base != READ_ONCE(timer->base->cpu_base));

 return false;
}
EXPORT_SYMBOL_GPL(hrtimer_active);
static void __run_hrtimer(struct hrtimer_cpu_base *cpu_base,
     struct hrtimer_clock_base *base,
     struct hrtimer *timer, ktime_t *now)
{
 enum hrtimer_restart (*fn)(struct hrtimer *);
 int restart;

 lockdep_assert_held(&cpu_base->lock);

 debug_deactivate(timer);
 cpu_base->running = timer;
 raw_write_seqcount_barrier(&cpu_base->seq);

 __remove_hrtimer(timer, base, HRTIMER_STATE_INACTIVE, 0);
 timer_stats_account_hrtimer(timer);
 fn = timer->function;






 if (IS_ENABLED(CONFIG_TIME_LOW_RES))
  timer->is_rel = false;






 raw_spin_unlock(&cpu_base->lock);
 trace_hrtimer_expire_entry(timer, now);
 restart = fn(timer);
 trace_hrtimer_expire_exit(timer);
 raw_spin_lock(&cpu_base->lock);
 if (restart != HRTIMER_NORESTART &&
     !(timer->state & HRTIMER_STATE_ENQUEUED))
  enqueue_hrtimer(timer, base);
 raw_write_seqcount_barrier(&cpu_base->seq);

 WARN_ON_ONCE(cpu_base->running != timer);
 cpu_base->running = NULL;
}

static void __hrtimer_run_queues(struct hrtimer_cpu_base *cpu_base, ktime_t now)
{
 struct hrtimer_clock_base *base = cpu_base->clock_base;
 unsigned int active = cpu_base->active_bases;

 for (; active; base++, active >>= 1) {
  struct timerqueue_node *node;
  ktime_t basenow;

  if (!(active & 0x01))
   continue;

  basenow = ktime_add(now, base->offset);

  while ((node = timerqueue_getnext(&base->active))) {
   struct hrtimer *timer;

   timer = container_of(node, struct hrtimer, node);
   if (basenow.tv64 < hrtimer_get_softexpires_tv64(timer))
    break;

   __run_hrtimer(cpu_base, base, timer, &basenow);
  }
 }
}






void hrtimer_interrupt(struct clock_event_device *dev)
{
 struct hrtimer_cpu_base *cpu_base = this_cpu_ptr(&hrtimer_bases);
 ktime_t expires_next, now, entry_time, delta;
 int retries = 0;

 BUG_ON(!cpu_base->hres_active);
 cpu_base->nr_events++;
 dev->next_event.tv64 = KTIME_MAX;

 raw_spin_lock(&cpu_base->lock);
 entry_time = now = hrtimer_update_base(cpu_base);
retry:
 cpu_base->in_hrtirq = 1;







 cpu_base->expires_next.tv64 = KTIME_MAX;

 __hrtimer_run_queues(cpu_base, now);


 expires_next = __hrtimer_get_next_event(cpu_base);




 cpu_base->expires_next = expires_next;
 cpu_base->in_hrtirq = 0;
 raw_spin_unlock(&cpu_base->lock);


 if (!tick_program_event(expires_next, 0)) {
  cpu_base->hang_detected = 0;
  return;
 }
 raw_spin_lock(&cpu_base->lock);
 now = hrtimer_update_base(cpu_base);
 cpu_base->nr_retries++;
 if (++retries < 3)
  goto retry;






 cpu_base->nr_hangs++;
 cpu_base->hang_detected = 1;
 raw_spin_unlock(&cpu_base->lock);
 delta = ktime_sub(now, entry_time);
 if ((unsigned int)delta.tv64 > cpu_base->max_hang_time)
  cpu_base->max_hang_time = (unsigned int) delta.tv64;




 if (delta.tv64 > 100 * NSEC_PER_MSEC)
  expires_next = ktime_add_ns(now, 100 * NSEC_PER_MSEC);
 else
  expires_next = ktime_add(now, delta);
 tick_program_event(expires_next, 1);
 printk_once(KERN_WARNING "hrtimer: interrupt took %llu ns\n",
      ktime_to_ns(delta));
}





static inline void __hrtimer_peek_ahead_timers(void)
{
 struct tick_device *td;

 if (!hrtimer_hres_active())
  return;

 td = this_cpu_ptr(&tick_cpu_device);
 if (td && td->evtdev)
  hrtimer_interrupt(td->evtdev);
}


static inline void __hrtimer_peek_ahead_timers(void) { }





void hrtimer_run_queues(void)
{
 struct hrtimer_cpu_base *cpu_base = this_cpu_ptr(&hrtimer_bases);
 ktime_t now;

 if (__hrtimer_hres_active(cpu_base))
  return;
 if (tick_check_oneshot_change(!hrtimer_is_hres_enabled())) {
  hrtimer_switch_to_hres();
  return;
 }

 raw_spin_lock(&cpu_base->lock);
 now = hrtimer_update_base(cpu_base);
 __hrtimer_run_queues(cpu_base, now);
 raw_spin_unlock(&cpu_base->lock);
}




static enum hrtimer_restart hrtimer_wakeup(struct hrtimer *timer)
{
 struct hrtimer_sleeper *t =
  container_of(timer, struct hrtimer_sleeper, timer);
 struct task_struct *task = t->task;

 t->task = NULL;
 if (task)
  wake_up_process(task);

 return HRTIMER_NORESTART;
}

void hrtimer_init_sleeper(struct hrtimer_sleeper *sl, struct task_struct *task)
{
 sl->timer.function = hrtimer_wakeup;
 sl->task = task;
}
EXPORT_SYMBOL_GPL(hrtimer_init_sleeper);

static int __sched do_nanosleep(struct hrtimer_sleeper *t, enum hrtimer_mode mode)
{
 hrtimer_init_sleeper(t, current);

 do {
  set_current_state(TASK_INTERRUPTIBLE);
  hrtimer_start_expires(&t->timer, mode);

  if (likely(t->task))
   freezable_schedule();

  hrtimer_cancel(&t->timer);
  mode = HRTIMER_MODE_ABS;

 } while (t->task && !signal_pending(current));

 __set_current_state(TASK_RUNNING);

 return t->task == NULL;
}

static int update_rmtp(struct hrtimer *timer, struct timespec __user *rmtp)
{
 struct timespec rmt;
 ktime_t rem;

 rem = hrtimer_expires_remaining(timer);
 if (rem.tv64 <= 0)
  return 0;
 rmt = ktime_to_timespec(rem);

 if (copy_to_user(rmtp, &rmt, sizeof(*rmtp)))
  return -EFAULT;

 return 1;
}

long __sched hrtimer_nanosleep_restart(struct restart_block *restart)
{
 struct hrtimer_sleeper t;
 struct timespec __user *rmtp;
 int ret = 0;

 hrtimer_init_on_stack(&t.timer, restart->nanosleep.clockid,
    HRTIMER_MODE_ABS);
 hrtimer_set_expires_tv64(&t.timer, restart->nanosleep.expires);

 if (do_nanosleep(&t, HRTIMER_MODE_ABS))
  goto out;

 rmtp = restart->nanosleep.rmtp;
 if (rmtp) {
  ret = update_rmtp(&t.timer, rmtp);
  if (ret <= 0)
   goto out;
 }


 ret = -ERESTART_RESTARTBLOCK;
out:
 destroy_hrtimer_on_stack(&t.timer);
 return ret;
}

long hrtimer_nanosleep(struct timespec *rqtp, struct timespec __user *rmtp,
         const enum hrtimer_mode mode, const clockid_t clockid)
{
 struct restart_block *restart;
 struct hrtimer_sleeper t;
 int ret = 0;
 u64 slack;

 slack = current->timer_slack_ns;
 if (dl_task(current) || rt_task(current))
  slack = 0;

 hrtimer_init_on_stack(&t.timer, clockid, mode);
 hrtimer_set_expires_range_ns(&t.timer, timespec_to_ktime(*rqtp), slack);
 if (do_nanosleep(&t, mode))
  goto out;


 if (mode == HRTIMER_MODE_ABS) {
  ret = -ERESTARTNOHAND;
  goto out;
 }

 if (rmtp) {
  ret = update_rmtp(&t.timer, rmtp);
  if (ret <= 0)
   goto out;
 }

 restart = &current->restart_block;
 restart->fn = hrtimer_nanosleep_restart;
 restart->nanosleep.clockid = t.timer.base->clockid;
 restart->nanosleep.rmtp = rmtp;
 restart->nanosleep.expires = hrtimer_get_expires_tv64(&t.timer);

 ret = -ERESTART_RESTARTBLOCK;
out:
 destroy_hrtimer_on_stack(&t.timer);
 return ret;
}

SYSCALL_DEFINE2(nanosleep, struct timespec __user *, rqtp,
  struct timespec __user *, rmtp)
{
 struct timespec tu;

 if (copy_from_user(&tu, rqtp, sizeof(tu)))
  return -EFAULT;

 if (!timespec_valid(&tu))
  return -EINVAL;

 return hrtimer_nanosleep(&tu, rmtp, HRTIMER_MODE_REL, CLOCK_MONOTONIC);
}




static void init_hrtimers_cpu(int cpu)
{
 struct hrtimer_cpu_base *cpu_base = &per_cpu(hrtimer_bases, cpu);
 int i;

 for (i = 0; i < HRTIMER_MAX_CLOCK_BASES; i++) {
  cpu_base->clock_base[i].cpu_base = cpu_base;
  timerqueue_init_head(&cpu_base->clock_base[i].active);
 }

 cpu_base->cpu = cpu;
 hrtimer_init_hres(cpu_base);
}


static void migrate_hrtimer_list(struct hrtimer_clock_base *old_base,
    struct hrtimer_clock_base *new_base)
{
 struct hrtimer *timer;
 struct timerqueue_node *node;

 while ((node = timerqueue_getnext(&old_base->active))) {
  timer = container_of(node, struct hrtimer, node);
  BUG_ON(hrtimer_callback_running(timer));
  debug_deactivate(timer);






  __remove_hrtimer(timer, old_base, HRTIMER_STATE_ENQUEUED, 0);
  timer->base = new_base;
  enqueue_hrtimer(timer, new_base);
 }
}

static void migrate_hrtimers(int scpu)
{
 struct hrtimer_cpu_base *old_base, *new_base;
 int i;

 BUG_ON(cpu_online(scpu));
 tick_cancel_sched_timer(scpu);

 local_irq_disable();
 old_base = &per_cpu(hrtimer_bases, scpu);
 new_base = this_cpu_ptr(&hrtimer_bases);




 raw_spin_lock(&new_base->lock);
 raw_spin_lock_nested(&old_base->lock, SINGLE_DEPTH_NESTING);

 for (i = 0; i < HRTIMER_MAX_CLOCK_BASES; i++) {
  migrate_hrtimer_list(&old_base->clock_base[i],
         &new_base->clock_base[i]);
 }

 raw_spin_unlock(&old_base->lock);
 raw_spin_unlock(&new_base->lock);


 __hrtimer_peek_ahead_timers();
 local_irq_enable();
}


static int hrtimer_cpu_notify(struct notifier_block *self,
     unsigned long action, void *hcpu)
{
 int scpu = (long)hcpu;

 switch (action) {

 case CPU_UP_PREPARE:
 case CPU_UP_PREPARE_FROZEN:
  init_hrtimers_cpu(scpu);
  break;

 case CPU_DEAD:
 case CPU_DEAD_FROZEN:
  migrate_hrtimers(scpu);
  break;

 default:
  break;
 }

 return NOTIFY_OK;
}

static struct notifier_block hrtimers_nb = {
 .notifier_call = hrtimer_cpu_notify,
};

void __init hrtimers_init(void)
{
 hrtimer_cpu_notify(&hrtimers_nb, (unsigned long)CPU_UP_PREPARE,
     (void *)(long)smp_processor_id());
 register_cpu_notifier(&hrtimers_nb);
}
int __sched
schedule_hrtimeout_range_clock(ktime_t *expires, u64 delta,
          const enum hrtimer_mode mode, int clock)
{
 struct hrtimer_sleeper t;





 if (expires && !expires->tv64) {
  __set_current_state(TASK_RUNNING);
  return 0;
 }




 if (!expires) {
  schedule();
  return -EINTR;
 }

 hrtimer_init_on_stack(&t.timer, clock, mode);
 hrtimer_set_expires_range_ns(&t.timer, *expires, delta);

 hrtimer_init_sleeper(&t, current);

 hrtimer_start_expires(&t.timer, mode);

 if (likely(t.task))
  schedule();

 hrtimer_cancel(&t.timer);
 destroy_hrtimer_on_stack(&t.timer);

 __set_current_state(TASK_RUNNING);

 return !t.task ? 0 : -EINTR;
}
int __sched schedule_hrtimeout_range(ktime_t *expires, u64 delta,
         const enum hrtimer_mode mode)
{
 return schedule_hrtimeout_range_clock(expires, delta, mode,
           CLOCK_MONOTONIC);
}
EXPORT_SYMBOL_GPL(schedule_hrtimeout_range);
int __sched schedule_hrtimeout(ktime_t *expires,
          const enum hrtimer_mode mode)
{
 return schedule_hrtimeout_range(expires, 0, mode);
}
EXPORT_SYMBOL_GPL(schedule_hrtimeout);











int __read_mostly sysctl_hung_task_check_count = PID_MAX_LIMIT;




unsigned long __read_mostly sysctl_hung_task_timeout_secs = CONFIG_DEFAULT_HUNG_TASK_TIMEOUT;

int __read_mostly sysctl_hung_task_warnings = 10;

static int __read_mostly did_panic;

static struct task_struct *watchdog_task;





unsigned int __read_mostly sysctl_hung_task_panic =
    CONFIG_BOOTPARAM_HUNG_TASK_PANIC_VALUE;

static int __init hung_task_panic_setup(char *str)
{
 int rc = kstrtouint(str, 0, &sysctl_hung_task_panic);

 if (rc)
  return rc;
 return 1;
}
__setup("hung_task_panic=", hung_task_panic_setup);

static int
hung_task_panic(struct notifier_block *this, unsigned long event, void *ptr)
{
 did_panic = 1;

 return NOTIFY_DONE;
}

static struct notifier_block panic_block = {
 .notifier_call = hung_task_panic,
};

static void check_hung_task(struct task_struct *t, unsigned long timeout)
{
 unsigned long switch_count = t->nvcsw + t->nivcsw;





 if (unlikely(t->flags & (PF_FROZEN | PF_FREEZER_SKIP)))
     return;






 if (unlikely(!switch_count))
  return;

 if (switch_count != t->last_switch_count) {
  t->last_switch_count = switch_count;
  return;
 }

 trace_sched_process_hang(t);

 if (!sysctl_hung_task_warnings)
  return;

 if (sysctl_hung_task_warnings > 0)
  sysctl_hung_task_warnings--;





 pr_err("INFO: task %s:%d blocked for more than %ld seconds.\n",
  t->comm, t->pid, timeout);
 pr_err("      %s %s %.*s\n",
  print_tainted(), init_utsname()->release,
  (int)strcspn(init_utsname()->version, " "),
  init_utsname()->version);
 pr_err("\"echo 0 > /proc/sys/kernel/hung_task_timeout_secs\""
  " disables this message.\n");
 sched_show_task(t);
 debug_show_held_locks(t);

 touch_nmi_watchdog();

 if (sysctl_hung_task_panic) {
  trigger_all_cpu_backtrace();
  panic("hung_task: blocked tasks");
 }
}
static bool rcu_lock_break(struct task_struct *g, struct task_struct *t)
{
 bool can_cont;

 get_task_struct(g);
 get_task_struct(t);
 rcu_read_unlock();
 cond_resched();
 rcu_read_lock();
 can_cont = pid_alive(g) && pid_alive(t);
 put_task_struct(t);
 put_task_struct(g);

 return can_cont;
}






static void check_hung_uninterruptible_tasks(unsigned long timeout)
{
 int max_count = sysctl_hung_task_check_count;
 int batch_count = HUNG_TASK_BATCHING;
 struct task_struct *g, *t;





 if (test_taint(TAINT_DIE) || did_panic)
  return;

 rcu_read_lock();
 for_each_process_thread(g, t) {
  if (!max_count--)
   goto unlock;
  if (!--batch_count) {
   batch_count = HUNG_TASK_BATCHING;
   if (!rcu_lock_break(g, t))
    goto unlock;
  }

  if (t->state == TASK_UNINTERRUPTIBLE)
   check_hung_task(t, timeout);
 }
 unlock:
 rcu_read_unlock();
}

static long hung_timeout_jiffies(unsigned long last_checked,
     unsigned long timeout)
{

 return timeout ? last_checked - jiffies + timeout * HZ :
  MAX_SCHEDULE_TIMEOUT;
}




int proc_dohung_task_timeout_secs(struct ctl_table *table, int write,
      void __user *buffer,
      size_t *lenp, loff_t *ppos)
{
 int ret;

 ret = proc_doulongvec_minmax(table, write, buffer, lenp, ppos);

 if (ret || !write)
  goto out;

 wake_up_process(watchdog_task);

 out:
 return ret;
}

static atomic_t reset_hung_task = ATOMIC_INIT(0);

void reset_hung_task_detector(void)
{
 atomic_set(&reset_hung_task, 1);
}
EXPORT_SYMBOL_GPL(reset_hung_task_detector);




static int watchdog(void *dummy)
{
 unsigned long hung_last_checked = jiffies;

 set_user_nice(current, 0);

 for ( ; ; ) {
  unsigned long timeout = sysctl_hung_task_timeout_secs;
  long t = hung_timeout_jiffies(hung_last_checked, timeout);

  if (t <= 0) {
   if (!atomic_xchg(&reset_hung_task, 0))
    check_hung_uninterruptible_tasks(timeout);
   hung_last_checked = jiffies;
   continue;
  }
  schedule_timeout_interruptible(t);
 }

 return 0;
}

static int __init hung_task_init(void)
{
 atomic_notifier_chain_register(&panic_notifier_list, &panic_block);
 watchdog_task = kthread_run(watchdog, NULL, "khungtaskd");

 return 0;
}
subsys_initcall(hung_task_init);




struct bp_cpuinfo {

 unsigned int cpu_pinned;

 unsigned int *tsk_pinned;

 unsigned int flexible;
};

static DEFINE_PER_CPU(struct bp_cpuinfo, bp_cpuinfo[TYPE_MAX]);
static int nr_slots[TYPE_MAX];

static struct bp_cpuinfo *get_bp_info(int cpu, enum bp_type_idx type)
{
 return per_cpu_ptr(bp_cpuinfo + type, cpu);
}


static LIST_HEAD(bp_task_head);

static int constraints_initialized;


struct bp_busy_slots {
 unsigned int pinned;
 unsigned int flexible;
};


static DEFINE_MUTEX(nr_bp_mutex);

__weak int hw_breakpoint_weight(struct perf_event *bp)
{
 return 1;
}

static inline enum bp_type_idx find_slot_idx(struct perf_event *bp)
{
 if (bp->attr.bp_type & HW_BREAKPOINT_RW)
  return TYPE_DATA;

 return TYPE_INST;
}





static unsigned int max_task_bp_pinned(int cpu, enum bp_type_idx type)
{
 unsigned int *tsk_pinned = get_bp_info(cpu, type)->tsk_pinned;
 int i;

 for (i = nr_slots[type] - 1; i >= 0; i--) {
  if (tsk_pinned[i] > 0)
   return i + 1;
 }

 return 0;
}





static int task_bp_pinned(int cpu, struct perf_event *bp, enum bp_type_idx type)
{
 struct task_struct *tsk = bp->hw.target;
 struct perf_event *iter;
 int count = 0;

 list_for_each_entry(iter, &bp_task_head, hw.bp_list) {
  if (iter->hw.target == tsk &&
      find_slot_idx(iter) == type &&
      (iter->cpu < 0 || cpu == iter->cpu))
   count += hw_breakpoint_weight(iter);
 }

 return count;
}

static const struct cpumask *cpumask_of_bp(struct perf_event *bp)
{
 if (bp->cpu >= 0)
  return cpumask_of(bp->cpu);
 return cpu_possible_mask;
}





static void
fetch_bp_busy_slots(struct bp_busy_slots *slots, struct perf_event *bp,
      enum bp_type_idx type)
{
 const struct cpumask *cpumask = cpumask_of_bp(bp);
 int cpu;

 for_each_cpu(cpu, cpumask) {
  struct bp_cpuinfo *info = get_bp_info(cpu, type);
  int nr;

  nr = info->cpu_pinned;
  if (!bp->hw.target)
   nr += max_task_bp_pinned(cpu, type);
  else
   nr += task_bp_pinned(cpu, bp, type);

  if (nr > slots->pinned)
   slots->pinned = nr;

  nr = info->flexible;
  if (nr > slots->flexible)
   slots->flexible = nr;
 }
}






static void
fetch_this_slot(struct bp_busy_slots *slots, int weight)
{
 slots->pinned += weight;
}




static void toggle_bp_task_slot(struct perf_event *bp, int cpu,
    enum bp_type_idx type, int weight)
{
 unsigned int *tsk_pinned = get_bp_info(cpu, type)->tsk_pinned;
 int old_idx, new_idx;

 old_idx = task_bp_pinned(cpu, bp, type) - 1;
 new_idx = old_idx + weight;

 if (old_idx >= 0)
  tsk_pinned[old_idx]--;
 if (new_idx >= 0)
  tsk_pinned[new_idx]++;
}




static void
toggle_bp_slot(struct perf_event *bp, bool enable, enum bp_type_idx type,
        int weight)
{
 const struct cpumask *cpumask = cpumask_of_bp(bp);
 int cpu;

 if (!enable)
  weight = -weight;


 if (!bp->hw.target) {
  get_bp_info(bp->cpu, type)->cpu_pinned += weight;
  return;
 }


 for_each_cpu(cpu, cpumask)
  toggle_bp_task_slot(bp, cpu, type, weight);

 if (enable)
  list_add_tail(&bp->hw.bp_list, &bp_task_head);
 else
  list_del(&bp->hw.bp_list);
}




__weak void arch_unregister_hw_breakpoint(struct perf_event *bp)
{




}
static int __reserve_bp_slot(struct perf_event *bp)
{
 struct bp_busy_slots slots = {0};
 enum bp_type_idx type;
 int weight;


 if (!constraints_initialized)
  return -ENOMEM;


 if (bp->attr.bp_type == HW_BREAKPOINT_EMPTY ||
     bp->attr.bp_type == HW_BREAKPOINT_INVALID)
  return -EINVAL;

 type = find_slot_idx(bp);
 weight = hw_breakpoint_weight(bp);

 fetch_bp_busy_slots(&slots, bp, type);




 fetch_this_slot(&slots, weight);


 if (slots.pinned + (!!slots.flexible) > nr_slots[type])
  return -ENOSPC;

 toggle_bp_slot(bp, true, type, weight);

 return 0;
}

int reserve_bp_slot(struct perf_event *bp)
{
 int ret;

 mutex_lock(&nr_bp_mutex);

 ret = __reserve_bp_slot(bp);

 mutex_unlock(&nr_bp_mutex);

 return ret;
}

static void __release_bp_slot(struct perf_event *bp)
{
 enum bp_type_idx type;
 int weight;

 type = find_slot_idx(bp);
 weight = hw_breakpoint_weight(bp);
 toggle_bp_slot(bp, false, type, weight);
}

void release_bp_slot(struct perf_event *bp)
{
 mutex_lock(&nr_bp_mutex);

 arch_unregister_hw_breakpoint(bp);
 __release_bp_slot(bp);

 mutex_unlock(&nr_bp_mutex);
}






int dbg_reserve_bp_slot(struct perf_event *bp)
{
 if (mutex_is_locked(&nr_bp_mutex))
  return -1;

 return __reserve_bp_slot(bp);
}

int dbg_release_bp_slot(struct perf_event *bp)
{
 if (mutex_is_locked(&nr_bp_mutex))
  return -1;

 __release_bp_slot(bp);

 return 0;
}

static int validate_hw_breakpoint(struct perf_event *bp)
{
 int ret;

 ret = arch_validate_hwbkpt_settings(bp);
 if (ret)
  return ret;

 if (arch_check_bp_in_kernelspace(bp)) {
  if (bp->attr.exclude_kernel)
   return -EINVAL;




  if (!capable(CAP_SYS_ADMIN))
   return -EPERM;
 }

 return 0;
}

int register_perf_hw_breakpoint(struct perf_event *bp)
{
 int ret;

 ret = reserve_bp_slot(bp);
 if (ret)
  return ret;

 ret = validate_hw_breakpoint(bp);


 if (ret)
  release_bp_slot(bp);

 return ret;
}







struct perf_event *
register_user_hw_breakpoint(struct perf_event_attr *attr,
       perf_overflow_handler_t triggered,
       void *context,
       struct task_struct *tsk)
{
 return perf_event_create_kernel_counter(attr, -1, tsk, triggered,
      context);
}
EXPORT_SYMBOL_GPL(register_user_hw_breakpoint);
int modify_user_hw_breakpoint(struct perf_event *bp, struct perf_event_attr *attr)
{
 u64 old_addr = bp->attr.bp_addr;
 u64 old_len = bp->attr.bp_len;
 int old_type = bp->attr.bp_type;
 int err = 0;







 if (irqs_disabled() && bp->ctx && bp->ctx->task == current)
  perf_event_disable_local(bp);
 else
  perf_event_disable(bp);

 bp->attr.bp_addr = attr->bp_addr;
 bp->attr.bp_type = attr->bp_type;
 bp->attr.bp_len = attr->bp_len;

 if (attr->disabled)
  goto end;

 err = validate_hw_breakpoint(bp);
 if (!err)
  perf_event_enable(bp);

 if (err) {
  bp->attr.bp_addr = old_addr;
  bp->attr.bp_type = old_type;
  bp->attr.bp_len = old_len;
  if (!bp->attr.disabled)
   perf_event_enable(bp);

  return err;
 }

end:
 bp->attr.disabled = attr->disabled;

 return 0;
}
EXPORT_SYMBOL_GPL(modify_user_hw_breakpoint);





void unregister_hw_breakpoint(struct perf_event *bp)
{
 if (!bp)
  return;
 perf_event_release_kernel(bp);
}
EXPORT_SYMBOL_GPL(unregister_hw_breakpoint);
struct perf_event * __percpu *
register_wide_hw_breakpoint(struct perf_event_attr *attr,
       perf_overflow_handler_t triggered,
       void *context)
{
 struct perf_event * __percpu *cpu_events, *bp;
 long err = 0;
 int cpu;

 cpu_events = alloc_percpu(typeof(*cpu_events));
 if (!cpu_events)
  return (void __percpu __force *)ERR_PTR(-ENOMEM);

 get_online_cpus();
 for_each_online_cpu(cpu) {
  bp = perf_event_create_kernel_counter(attr, cpu, NULL,
            triggered, context);
  if (IS_ERR(bp)) {
   err = PTR_ERR(bp);
   break;
  }

  per_cpu(*cpu_events, cpu) = bp;
 }
 put_online_cpus();

 if (likely(!err))
  return cpu_events;

 unregister_wide_hw_breakpoint(cpu_events);
 return (void __percpu __force *)ERR_PTR(err);
}
EXPORT_SYMBOL_GPL(register_wide_hw_breakpoint);





void unregister_wide_hw_breakpoint(struct perf_event * __percpu *cpu_events)
{
 int cpu;

 for_each_possible_cpu(cpu)
  unregister_hw_breakpoint(per_cpu(*cpu_events, cpu));

 free_percpu(cpu_events);
}
EXPORT_SYMBOL_GPL(unregister_wide_hw_breakpoint);

static struct notifier_block hw_breakpoint_exceptions_nb = {
 .notifier_call = hw_breakpoint_exceptions_notify,

 .priority = 0x7fffffff
};

static void bp_perf_event_destroy(struct perf_event *event)
{
 release_bp_slot(event);
}

static int hw_breakpoint_event_init(struct perf_event *bp)
{
 int err;

 if (bp->attr.type != PERF_TYPE_BREAKPOINT)
  return -ENOENT;




 if (has_branch_stack(bp))
  return -EOPNOTSUPP;

 err = register_perf_hw_breakpoint(bp);
 if (err)
  return err;

 bp->destroy = bp_perf_event_destroy;

 return 0;
}

static int hw_breakpoint_add(struct perf_event *bp, int flags)
{
 if (!(flags & PERF_EF_START))
  bp->hw.state = PERF_HES_STOPPED;

 if (is_sampling_event(bp)) {
  bp->hw.last_period = bp->hw.sample_period;
  perf_swevent_set_period(bp);
 }

 return arch_install_hw_breakpoint(bp);
}

static void hw_breakpoint_del(struct perf_event *bp, int flags)
{
 arch_uninstall_hw_breakpoint(bp);
}

static void hw_breakpoint_start(struct perf_event *bp, int flags)
{
 bp->hw.state = 0;
}

static void hw_breakpoint_stop(struct perf_event *bp, int flags)
{
 bp->hw.state = PERF_HES_STOPPED;
}

static struct pmu perf_breakpoint = {
 .task_ctx_nr = perf_sw_context,

 .event_init = hw_breakpoint_event_init,
 .add = hw_breakpoint_add,
 .del = hw_breakpoint_del,
 .start = hw_breakpoint_start,
 .stop = hw_breakpoint_stop,
 .read = hw_breakpoint_pmu_read,
};

int __init init_hw_breakpoint(void)
{
 int cpu, err_cpu;
 int i;

 for (i = 0; i < TYPE_MAX; i++)
  nr_slots[i] = hw_breakpoint_slots(i);

 for_each_possible_cpu(cpu) {
  for (i = 0; i < TYPE_MAX; i++) {
   struct bp_cpuinfo *info = get_bp_info(cpu, i);

   info->tsk_pinned = kcalloc(nr_slots[i], sizeof(int),
       GFP_KERNEL);
   if (!info->tsk_pinned)
    goto err_alloc;
  }
 }

 constraints_initialized = 1;

 perf_pmu_register(&perf_breakpoint, "breakpoint", PERF_TYPE_BREAKPOINT);

 return register_die_notifier(&hw_breakpoint_exceptions_nb);

 err_alloc:
 for_each_possible_cpu(err_cpu) {
  for (i = 0; i < TYPE_MAX; i++)
   kfree(get_bp_info(err_cpu, i)->tsk_pinned);
  if (err_cpu == cpu)
   break;
 }

 return -ENOMEM;
}

enum bpf_type {
 BPF_TYPE_UNSPEC = 0,
 BPF_TYPE_PROG,
 BPF_TYPE_MAP,
};

static void *bpf_any_get(void *raw, enum bpf_type type)
{
 switch (type) {
 case BPF_TYPE_PROG:
  raw = bpf_prog_inc(raw);
  break;
 case BPF_TYPE_MAP:
  raw = bpf_map_inc(raw, true);
  break;
 default:
  WARN_ON_ONCE(1);
  break;
 }

 return raw;
}

static void bpf_any_put(void *raw, enum bpf_type type)
{
 switch (type) {
 case BPF_TYPE_PROG:
  bpf_prog_put(raw);
  break;
 case BPF_TYPE_MAP:
  bpf_map_put_with_uref(raw);
  break;
 default:
  WARN_ON_ONCE(1);
  break;
 }
}

static void *bpf_fd_probe_obj(u32 ufd, enum bpf_type *type)
{
 void *raw;

 *type = BPF_TYPE_MAP;
 raw = bpf_map_get_with_uref(ufd);
 if (IS_ERR(raw)) {
  *type = BPF_TYPE_PROG;
  raw = bpf_prog_get(ufd);
 }

 return raw;
}

static const struct inode_operations bpf_dir_iops;

static const struct inode_operations bpf_prog_iops = { };
static const struct inode_operations bpf_map_iops = { };

static struct inode *bpf_get_inode(struct super_block *sb,
       const struct inode *dir,
       umode_t mode)
{
 struct inode *inode;

 switch (mode & S_IFMT) {
 case S_IFDIR:
 case S_IFREG:
  break;
 default:
  return ERR_PTR(-EINVAL);
 }

 inode = new_inode(sb);
 if (!inode)
  return ERR_PTR(-ENOSPC);

 inode->i_ino = get_next_ino();
 inode->i_atime = CURRENT_TIME;
 inode->i_mtime = inode->i_atime;
 inode->i_ctime = inode->i_atime;

 inode_init_owner(inode, dir, mode);

 return inode;
}

static int bpf_inode_type(const struct inode *inode, enum bpf_type *type)
{
 *type = BPF_TYPE_UNSPEC;
 if (inode->i_op == &bpf_prog_iops)
  *type = BPF_TYPE_PROG;
 else if (inode->i_op == &bpf_map_iops)
  *type = BPF_TYPE_MAP;
 else
  return -EACCES;

 return 0;
}

static int bpf_mkdir(struct inode *dir, struct dentry *dentry, umode_t mode)
{
 struct inode *inode;

 inode = bpf_get_inode(dir->i_sb, dir, mode | S_IFDIR);
 if (IS_ERR(inode))
  return PTR_ERR(inode);

 inode->i_op = &bpf_dir_iops;
 inode->i_fop = &simple_dir_operations;

 inc_nlink(inode);
 inc_nlink(dir);

 d_instantiate(dentry, inode);
 dget(dentry);

 return 0;
}

static int bpf_mkobj_ops(struct inode *dir, struct dentry *dentry,
    umode_t mode, const struct inode_operations *iops)
{
 struct inode *inode;

 inode = bpf_get_inode(dir->i_sb, dir, mode | S_IFREG);
 if (IS_ERR(inode))
  return PTR_ERR(inode);

 inode->i_op = iops;
 inode->i_private = dentry->d_fsdata;

 d_instantiate(dentry, inode);
 dget(dentry);

 return 0;
}

static int bpf_mkobj(struct inode *dir, struct dentry *dentry, umode_t mode,
       dev_t devt)
{
 enum bpf_type type = MINOR(devt);

 if (MAJOR(devt) != UNNAMED_MAJOR || !S_ISREG(mode) ||
     dentry->d_fsdata == NULL)
  return -EPERM;

 switch (type) {
 case BPF_TYPE_PROG:
  return bpf_mkobj_ops(dir, dentry, mode, &bpf_prog_iops);
 case BPF_TYPE_MAP:
  return bpf_mkobj_ops(dir, dentry, mode, &bpf_map_iops);
 default:
  return -EPERM;
 }
}

static struct dentry *
bpf_lookup(struct inode *dir, struct dentry *dentry, unsigned flags)
{
 if (strchr(dentry->d_name.name, '.'))
  return ERR_PTR(-EPERM);
 return simple_lookup(dir, dentry, flags);
}

static const struct inode_operations bpf_dir_iops = {
 .lookup = bpf_lookup,
 .mknod = bpf_mkobj,
 .mkdir = bpf_mkdir,
 .rmdir = simple_rmdir,
 .rename = simple_rename,
 .link = simple_link,
 .unlink = simple_unlink,
};

static int bpf_obj_do_pin(const struct filename *pathname, void *raw,
     enum bpf_type type)
{
 struct dentry *dentry;
 struct inode *dir;
 struct path path;
 umode_t mode;
 dev_t devt;
 int ret;

 dentry = kern_path_create(AT_FDCWD, pathname->name, &path, 0);
 if (IS_ERR(dentry))
  return PTR_ERR(dentry);

 mode = S_IFREG | ((S_IRUSR | S_IWUSR) & ~current_umask());
 devt = MKDEV(UNNAMED_MAJOR, type);

 ret = security_path_mknod(&path, dentry, mode, devt);
 if (ret)
  goto out;

 dir = d_inode(path.dentry);
 if (dir->i_op != &bpf_dir_iops) {
  ret = -EPERM;
  goto out;
 }

 dentry->d_fsdata = raw;
 ret = vfs_mknod(dir, dentry, mode, devt);
 dentry->d_fsdata = NULL;
out:
 done_path_create(&path, dentry);
 return ret;
}

int bpf_obj_pin_user(u32 ufd, const char __user *pathname)
{
 struct filename *pname;
 enum bpf_type type;
 void *raw;
 int ret;

 pname = getname(pathname);
 if (IS_ERR(pname))
  return PTR_ERR(pname);

 raw = bpf_fd_probe_obj(ufd, &type);
 if (IS_ERR(raw)) {
  ret = PTR_ERR(raw);
  goto out;
 }

 ret = bpf_obj_do_pin(pname, raw, type);
 if (ret != 0)
  bpf_any_put(raw, type);
out:
 putname(pname);
 return ret;
}

static void *bpf_obj_do_get(const struct filename *pathname,
       enum bpf_type *type)
{
 struct inode *inode;
 struct path path;
 void *raw;
 int ret;

 ret = kern_path(pathname->name, LOOKUP_FOLLOW, &path);
 if (ret)
  return ERR_PTR(ret);

 inode = d_backing_inode(path.dentry);
 ret = inode_permission(inode, MAY_WRITE);
 if (ret)
  goto out;

 ret = bpf_inode_type(inode, type);
 if (ret)
  goto out;

 raw = bpf_any_get(inode->i_private, *type);
 if (!IS_ERR(raw))
  touch_atime(&path);

 path_put(&path);
 return raw;
out:
 path_put(&path);
 return ERR_PTR(ret);
}

int bpf_obj_get_user(const char __user *pathname)
{
 enum bpf_type type = BPF_TYPE_UNSPEC;
 struct filename *pname;
 int ret = -ENOENT;
 void *raw;

 pname = getname(pathname);
 if (IS_ERR(pname))
  return PTR_ERR(pname);

 raw = bpf_obj_do_get(pname, &type);
 if (IS_ERR(raw)) {
  ret = PTR_ERR(raw);
  goto out;
 }

 if (type == BPF_TYPE_PROG)
  ret = bpf_prog_new_fd(raw);
 else if (type == BPF_TYPE_MAP)
  ret = bpf_map_new_fd(raw);
 else
  goto out;

 if (ret < 0)
  bpf_any_put(raw, type);
out:
 putname(pname);
 return ret;
}

static void bpf_evict_inode(struct inode *inode)
{
 enum bpf_type type;

 truncate_inode_pages_final(&inode->i_data);
 clear_inode(inode);

 if (!bpf_inode_type(inode, &type))
  bpf_any_put(inode->i_private, type);
}

static const struct super_operations bpf_super_ops = {
 .statfs = simple_statfs,
 .drop_inode = generic_delete_inode,
 .evict_inode = bpf_evict_inode,
};

static int bpf_fill_super(struct super_block *sb, void *data, int silent)
{
 static struct tree_descr bpf_rfiles[] = { { "" } };
 struct inode *inode;
 int ret;

 ret = simple_fill_super(sb, BPF_FS_MAGIC, bpf_rfiles);
 if (ret)
  return ret;

 sb->s_op = &bpf_super_ops;

 inode = sb->s_root->d_inode;
 inode->i_op = &bpf_dir_iops;
 inode->i_mode &= ~S_IALLUGO;
 inode->i_mode |= S_ISVTX | S_IRWXUGO;

 return 0;
}

static struct dentry *bpf_mount(struct file_system_type *type, int flags,
    const char *dev_name, void *data)
{
 return mount_nodev(type, flags, data, bpf_fill_super);
}

static struct file_system_type bpf_fs_type = {
 .owner = THIS_MODULE,
 .name = "bpf",
 .mount = bpf_mount,
 .kill_sb = kill_litter_super,
};

MODULE_ALIAS_FS("bpf");

static int __init bpf_init(void)
{
 int ret;

 ret = sysfs_create_mount_point(fs_kobj, "bpf");
 if (ret)
  return ret;

 ret = register_filesystem(&bpf_fs_type);
 if (ret)
  sysfs_remove_mount_point(fs_kobj, "bpf");

 return ret;
}
fs_initcall(bpf_init);

int irq_reserve_ipi(struct irq_domain *domain,
        const struct cpumask *dest)
{
 unsigned int nr_irqs, offset;
 struct irq_data *data;
 int virq, i;

 if (!domain ||!irq_domain_is_ipi(domain)) {
  pr_warn("Reservation on a non IPI domain\n");
  return -EINVAL;
 }

 if (!cpumask_subset(dest, cpu_possible_mask)) {
  pr_warn("Reservation is not in possible_cpu_mask\n");
  return -EINVAL;
 }

 nr_irqs = cpumask_weight(dest);
 if (!nr_irqs) {
  pr_warn("Reservation for empty destination mask\n");
  return -EINVAL;
 }

 if (irq_domain_is_ipi_single(domain)) {






  nr_irqs = 1;
  offset = 0;
 } else {
  unsigned int next;







  offset = cpumask_first(dest);




  next = cpumask_next_zero(offset, dest);
  if (next < nr_cpu_ids)
   next = cpumask_next(next, dest);
  if (next < nr_cpu_ids) {
   pr_warn("Destination mask has holes\n");
   return -EINVAL;
  }
 }

 virq = irq_domain_alloc_descs(-1, nr_irqs, 0, NUMA_NO_NODE);
 if (virq <= 0) {
  pr_warn("Can't reserve IPI, failed to alloc descs\n");
  return -ENOMEM;
 }

 virq = __irq_domain_alloc_irqs(domain, virq, nr_irqs, NUMA_NO_NODE,
           (void *) dest, true);

 if (virq <= 0) {
  pr_warn("Can't reserve IPI, failed to alloc hw irqs\n");
  goto free_descs;
 }

 for (i = 0; i < nr_irqs; i++) {
  data = irq_get_irq_data(virq + i);
  cpumask_copy(data->common->affinity, dest);
  data->common->ipi_offset = offset;
  irq_set_status_flags(virq + i, IRQ_NO_BALANCING);
 }
 return virq;

free_descs:
 irq_free_descs(virq, nr_irqs);
 return -EBUSY;
}
int irq_destroy_ipi(unsigned int irq, const struct cpumask *dest)
{
 struct irq_data *data = irq_get_irq_data(irq);
 struct cpumask *ipimask = data ? irq_data_get_affinity_mask(data) : NULL;
 struct irq_domain *domain;
 unsigned int nr_irqs;

 if (!irq || !data || !ipimask)
  return -EINVAL;

 domain = data->domain;
 if (WARN_ON(domain == NULL))
  return -EINVAL;

 if (!irq_domain_is_ipi(domain)) {
  pr_warn("Trying to destroy a non IPI domain!\n");
  return -EINVAL;
 }

 if (WARN_ON(!cpumask_subset(dest, ipimask)))




  return -EINVAL;

 if (irq_domain_is_ipi_per_cpu(domain)) {
  irq = irq + cpumask_first(dest) - data->common->ipi_offset;
  nr_irqs = cpumask_weight(dest);
 } else {
  nr_irqs = 1;
 }

 irq_domain_free_irqs(irq, nr_irqs);
 return 0;
}
irq_hw_number_t ipi_get_hwirq(unsigned int irq, unsigned int cpu)
{
 struct irq_data *data = irq_get_irq_data(irq);
 struct cpumask *ipimask = data ? irq_data_get_affinity_mask(data) : NULL;

 if (!data || !ipimask || cpu > nr_cpu_ids)
  return INVALID_HWIRQ;

 if (!cpumask_test_cpu(cpu, ipimask))
  return INVALID_HWIRQ;







 if (irq_domain_is_ipi_per_cpu(data->domain))
  data = irq_get_irq_data(irq + cpu - data->common->ipi_offset);

 return data ? irqd_to_hwirq(data) : INVALID_HWIRQ;
}
EXPORT_SYMBOL_GPL(ipi_get_hwirq);

static int ipi_send_verify(struct irq_chip *chip, struct irq_data *data,
      const struct cpumask *dest, unsigned int cpu)
{
 struct cpumask *ipimask = irq_data_get_affinity_mask(data);

 if (!chip || !ipimask)
  return -EINVAL;

 if (!chip->ipi_send_single && !chip->ipi_send_mask)
  return -EINVAL;

 if (cpu > nr_cpu_ids)
  return -EINVAL;

 if (dest) {
  if (!cpumask_subset(dest, ipimask))
   return -EINVAL;
 } else {
  if (!cpumask_test_cpu(cpu, ipimask))
   return -EINVAL;
 }
 return 0;
}
int __ipi_send_single(struct irq_desc *desc, unsigned int cpu)
{
 struct irq_data *data = irq_desc_get_irq_data(desc);
 struct irq_chip *chip = irq_data_get_irq_chip(data);






 if (WARN_ON_ONCE(ipi_send_verify(chip, data, NULL, cpu)))
  return -EINVAL;
 if (!chip->ipi_send_single) {
  chip->ipi_send_mask(data, cpumask_of(cpu));
  return 0;
 }


 if (irq_domain_is_ipi_per_cpu(data->domain) &&
     cpu != data->common->ipi_offset) {

  unsigned irq = data->irq + cpu - data->common->ipi_offset;

  data = irq_get_irq_data(irq);
 }
 chip->ipi_send_single(data, cpu);
 return 0;
}
int __ipi_send_mask(struct irq_desc *desc, const struct cpumask *dest)
{
 struct irq_data *data = irq_desc_get_irq_data(desc);
 struct irq_chip *chip = irq_data_get_irq_chip(data);
 unsigned int cpu;






 if (WARN_ON_ONCE(ipi_send_verify(chip, data, dest, 0)))
  return -EINVAL;
 if (chip->ipi_send_mask) {
  chip->ipi_send_mask(data, dest);
  return 0;
 }

 if (irq_domain_is_ipi_per_cpu(data->domain)) {
  unsigned int base = data->irq;

  for_each_cpu(cpu, dest) {
   unsigned irq = base + cpu - data->common->ipi_offset;

   data = irq_get_irq_data(irq);
   chip->ipi_send_single(data, cpu);
  }
 } else {
  for_each_cpu(cpu, dest)
   chip->ipi_send_single(data, cpu);
 }
 return 0;
}
int ipi_send_single(unsigned int virq, unsigned int cpu)
{
 struct irq_desc *desc = irq_to_desc(virq);
 struct irq_data *data = desc ? irq_desc_get_irq_data(desc) : NULL;
 struct irq_chip *chip = data ? irq_data_get_irq_chip(data) : NULL;

 if (WARN_ON_ONCE(ipi_send_verify(chip, data, NULL, cpu)))
  return -EINVAL;

 return __ipi_send_single(desc, cpu);
}
EXPORT_SYMBOL_GPL(ipi_send_single);
int ipi_send_mask(unsigned int virq, const struct cpumask *dest)
{
 struct irq_desc *desc = irq_to_desc(virq);
 struct irq_data *data = desc ? irq_desc_get_irq_data(desc) : NULL;
 struct irq_chip *chip = data ? irq_data_get_irq_chip(data) : NULL;

 if (WARN_ON_ONCE(ipi_send_verify(chip, data, dest, 0)))
  return -EINVAL;

 return __ipi_send_mask(desc, dest);
}
EXPORT_SYMBOL_GPL(ipi_send_mask);





static struct lock_class_key irq_desc_lock_class;

static int __init irq_affinity_setup(char *str)
{
 zalloc_cpumask_var(&irq_default_affinity, GFP_NOWAIT);
 cpulist_parse(str, irq_default_affinity);




 cpumask_set_cpu(smp_processor_id(), irq_default_affinity);
 return 1;
}
__setup("irqaffinity=", irq_affinity_setup);

static void __init init_irq_default_affinity(void)
{
 if (!irq_default_affinity)
  zalloc_cpumask_var(&irq_default_affinity, GFP_NOWAIT);
 if (cpumask_empty(irq_default_affinity))
  cpumask_setall(irq_default_affinity);
}
static void __init init_irq_default_affinity(void)
{
}

static int alloc_masks(struct irq_desc *desc, gfp_t gfp, int node)
{
 if (!zalloc_cpumask_var_node(&desc->irq_common_data.affinity,
         gfp, node))
  return -ENOMEM;

 if (!zalloc_cpumask_var_node(&desc->pending_mask, gfp, node)) {
  free_cpumask_var(desc->irq_common_data.affinity);
  return -ENOMEM;
 }
 return 0;
}

static void desc_smp_init(struct irq_desc *desc, int node)
{
 cpumask_copy(desc->irq_common_data.affinity, irq_default_affinity);
 cpumask_clear(desc->pending_mask);
 desc->irq_common_data.node = node;
}

static inline int
alloc_masks(struct irq_desc *desc, gfp_t gfp, int node) { return 0; }
static inline void desc_smp_init(struct irq_desc *desc, int node) { }

static void desc_set_defaults(unsigned int irq, struct irq_desc *desc, int node,
  struct module *owner)
{
 int cpu;

 desc->irq_common_data.handler_data = NULL;
 desc->irq_common_data.msi_desc = NULL;

 desc->irq_data.common = &desc->irq_common_data;
 desc->irq_data.irq = irq;
 desc->irq_data.chip = &no_irq_chip;
 desc->irq_data.chip_data = NULL;
 irq_settings_clr_and_set(desc, ~0, _IRQ_DEFAULT_INIT_FLAGS);
 irqd_set(&desc->irq_data, IRQD_IRQ_DISABLED);
 desc->handle_irq = handle_bad_irq;
 desc->depth = 1;
 desc->irq_count = 0;
 desc->irqs_unhandled = 0;
 desc->name = NULL;
 desc->owner = owner;
 for_each_possible_cpu(cpu)
  *per_cpu_ptr(desc->kstat_irqs, cpu) = 0;
 desc_smp_init(desc, node);
}

int nr_irqs = NR_IRQS;
EXPORT_SYMBOL_GPL(nr_irqs);

static DEFINE_MUTEX(sparse_irq_lock);
static DECLARE_BITMAP(allocated_irqs, IRQ_BITMAP_BITS);


static RADIX_TREE(irq_desc_tree, GFP_KERNEL);

static void irq_insert_desc(unsigned int irq, struct irq_desc *desc)
{
 radix_tree_insert(&irq_desc_tree, irq, desc);
}

struct irq_desc *irq_to_desc(unsigned int irq)
{
 return radix_tree_lookup(&irq_desc_tree, irq);
}
EXPORT_SYMBOL(irq_to_desc);

static void delete_irq_desc(unsigned int irq)
{
 radix_tree_delete(&irq_desc_tree, irq);
}

static void free_masks(struct irq_desc *desc)
{
 free_cpumask_var(desc->pending_mask);
 free_cpumask_var(desc->irq_common_data.affinity);
}
static inline void free_masks(struct irq_desc *desc) { }

void irq_lock_sparse(void)
{
 mutex_lock(&sparse_irq_lock);
}

void irq_unlock_sparse(void)
{
 mutex_unlock(&sparse_irq_lock);
}

static struct irq_desc *alloc_desc(int irq, int node, struct module *owner)
{
 struct irq_desc *desc;
 gfp_t gfp = GFP_KERNEL;

 desc = kzalloc_node(sizeof(*desc), gfp, node);
 if (!desc)
  return NULL;

 desc->kstat_irqs = alloc_percpu(unsigned int);
 if (!desc->kstat_irqs)
  goto err_desc;

 if (alloc_masks(desc, gfp, node))
  goto err_kstat;

 raw_spin_lock_init(&desc->lock);
 lockdep_set_class(&desc->lock, &irq_desc_lock_class);
 init_rcu_head(&desc->rcu);

 desc_set_defaults(irq, desc, node, owner);

 return desc;

err_kstat:
 free_percpu(desc->kstat_irqs);
err_desc:
 kfree(desc);
 return NULL;
}

static void delayed_free_desc(struct rcu_head *rhp)
{
 struct irq_desc *desc = container_of(rhp, struct irq_desc, rcu);

 free_masks(desc);
 free_percpu(desc->kstat_irqs);
 kfree(desc);
}

static void free_desc(unsigned int irq)
{
 struct irq_desc *desc = irq_to_desc(irq);

 unregister_irq_proc(irq, desc);







 mutex_lock(&sparse_irq_lock);
 delete_irq_desc(irq);
 mutex_unlock(&sparse_irq_lock);






 call_rcu(&desc->rcu, delayed_free_desc);
}

static int alloc_descs(unsigned int start, unsigned int cnt, int node,
         struct module *owner)
{
 struct irq_desc *desc;
 int i;

 for (i = 0; i < cnt; i++) {
  desc = alloc_desc(start + i, node, owner);
  if (!desc)
   goto err;
  mutex_lock(&sparse_irq_lock);
  irq_insert_desc(start + i, desc);
  mutex_unlock(&sparse_irq_lock);
 }
 return start;

err:
 for (i--; i >= 0; i--)
  free_desc(start + i);

 mutex_lock(&sparse_irq_lock);
 bitmap_clear(allocated_irqs, start, cnt);
 mutex_unlock(&sparse_irq_lock);
 return -ENOMEM;
}

static int irq_expand_nr_irqs(unsigned int nr)
{
 if (nr > IRQ_BITMAP_BITS)
  return -ENOMEM;
 nr_irqs = nr;
 return 0;
}

int __init early_irq_init(void)
{
 int i, initcnt, node = first_online_node;
 struct irq_desc *desc;

 init_irq_default_affinity();


 initcnt = arch_probe_nr_irqs();
 printk(KERN_INFO "NR_IRQS:%d nr_irqs:%d %d\n", NR_IRQS, nr_irqs, initcnt);

 if (WARN_ON(nr_irqs > IRQ_BITMAP_BITS))
  nr_irqs = IRQ_BITMAP_BITS;

 if (WARN_ON(initcnt > IRQ_BITMAP_BITS))
  initcnt = IRQ_BITMAP_BITS;

 if (initcnt > nr_irqs)
  nr_irqs = initcnt;

 for (i = 0; i < initcnt; i++) {
  desc = alloc_desc(i, node, NULL);
  set_bit(i, allocated_irqs);
  irq_insert_desc(i, desc);
 }
 return arch_early_irq_init();
}


struct irq_desc irq_desc[NR_IRQS] __cacheline_aligned_in_smp = {
 [0 ... NR_IRQS-1] = {
  .handle_irq = handle_bad_irq,
  .depth = 1,
  .lock = __RAW_SPIN_LOCK_UNLOCKED(irq_desc->lock),
 }
};

int __init early_irq_init(void)
{
 int count, i, node = first_online_node;
 struct irq_desc *desc;

 init_irq_default_affinity();

 printk(KERN_INFO "NR_IRQS:%d\n", NR_IRQS);

 desc = irq_desc;
 count = ARRAY_SIZE(irq_desc);

 for (i = 0; i < count; i++) {
  desc[i].kstat_irqs = alloc_percpu(unsigned int);
  alloc_masks(&desc[i], GFP_KERNEL, node);
  raw_spin_lock_init(&desc[i].lock);
  lockdep_set_class(&desc[i].lock, &irq_desc_lock_class);
  desc_set_defaults(i, &desc[i], node, NULL);
 }
 return arch_early_irq_init();
}

struct irq_desc *irq_to_desc(unsigned int irq)
{
 return (irq < NR_IRQS) ? irq_desc + irq : NULL;
}
EXPORT_SYMBOL(irq_to_desc);

static void free_desc(unsigned int irq)
{
 struct irq_desc *desc = irq_to_desc(irq);
 unsigned long flags;

 raw_spin_lock_irqsave(&desc->lock, flags);
 desc_set_defaults(irq, desc, irq_desc_get_node(desc), NULL);
 raw_spin_unlock_irqrestore(&desc->lock, flags);
}

static inline int alloc_descs(unsigned int start, unsigned int cnt, int node,
         struct module *owner)
{
 u32 i;

 for (i = 0; i < cnt; i++) {
  struct irq_desc *desc = irq_to_desc(start + i);

  desc->owner = owner;
 }
 return start;
}

static int irq_expand_nr_irqs(unsigned int nr)
{
 return -ENOMEM;
}

void irq_mark_irq(unsigned int irq)
{
 mutex_lock(&sparse_irq_lock);
 bitmap_set(allocated_irqs, irq, 1);
 mutex_unlock(&sparse_irq_lock);
}

void irq_init_desc(unsigned int irq)
{
 free_desc(irq);
}







int generic_handle_irq(unsigned int irq)
{
 struct irq_desc *desc = irq_to_desc(irq);

 if (!desc)
  return -EINVAL;
 generic_handle_irq_desc(desc);
 return 0;
}
EXPORT_SYMBOL_GPL(generic_handle_irq);

int __handle_domain_irq(struct irq_domain *domain, unsigned int hwirq,
   bool lookup, struct pt_regs *regs)
{
 struct pt_regs *old_regs = set_irq_regs(regs);
 unsigned int irq = hwirq;
 int ret = 0;

 irq_enter();

 if (lookup)
  irq = irq_find_mapping(domain, hwirq);





 if (unlikely(!irq || irq >= nr_irqs)) {
  ack_bad_irq(irq);
  ret = -EINVAL;
 } else {
  generic_handle_irq(irq);
 }

 irq_exit();
 set_irq_regs(old_regs);
 return ret;
}
void irq_free_descs(unsigned int from, unsigned int cnt)
{
 int i;

 if (from >= nr_irqs || (from + cnt) > nr_irqs)
  return;

 for (i = 0; i < cnt; i++)
  free_desc(from + i);

 mutex_lock(&sparse_irq_lock);
 bitmap_clear(allocated_irqs, from, cnt);
 mutex_unlock(&sparse_irq_lock);
}
EXPORT_SYMBOL_GPL(irq_free_descs);
int __ref
__irq_alloc_descs(int irq, unsigned int from, unsigned int cnt, int node,
    struct module *owner)
{
 int start, ret;

 if (!cnt)
  return -EINVAL;

 if (irq >= 0) {
  if (from > irq)
   return -EINVAL;
  from = irq;
 } else {





  from = arch_dynirq_lower_bound(from);
 }

 mutex_lock(&sparse_irq_lock);

 start = bitmap_find_next_zero_area(allocated_irqs, IRQ_BITMAP_BITS,
        from, cnt, 0);
 ret = -EEXIST;
 if (irq >=0 && start != irq)
  goto err;

 if (start + cnt > nr_irqs) {
  ret = irq_expand_nr_irqs(start + cnt);
  if (ret)
   goto err;
 }

 bitmap_set(allocated_irqs, start, cnt);
 mutex_unlock(&sparse_irq_lock);
 return alloc_descs(start, cnt, node, owner);

err:
 mutex_unlock(&sparse_irq_lock);
 return ret;
}
EXPORT_SYMBOL_GPL(__irq_alloc_descs);








unsigned int irq_alloc_hwirqs(int cnt, int node)
{
 int i, irq = __irq_alloc_descs(-1, 0, cnt, node, NULL);

 if (irq < 0)
  return 0;

 for (i = irq; cnt > 0; i++, cnt--) {
  if (arch_setup_hwirq(i, node))
   goto err;
  irq_clear_status_flags(i, _IRQ_NOREQUEST);
 }
 return irq;

err:
 for (i--; i >= irq; i--) {
  irq_set_status_flags(i, _IRQ_NOREQUEST | _IRQ_NOPROBE);
  arch_teardown_hwirq(i);
 }
 irq_free_descs(irq, cnt);
 return 0;
}
EXPORT_SYMBOL_GPL(irq_alloc_hwirqs);







void irq_free_hwirqs(unsigned int from, int cnt)
{
 int i, j;

 for (i = from, j = cnt; j > 0; i++, j--) {
  irq_set_status_flags(i, _IRQ_NOREQUEST | _IRQ_NOPROBE);
  arch_teardown_hwirq(i);
 }
 irq_free_descs(from, cnt);
}
EXPORT_SYMBOL_GPL(irq_free_hwirqs);







unsigned int irq_get_next_irq(unsigned int offset)
{
 return find_next_bit(allocated_irqs, nr_irqs, offset);
}

struct irq_desc *
__irq_get_desc_lock(unsigned int irq, unsigned long *flags, bool bus,
      unsigned int check)
{
 struct irq_desc *desc = irq_to_desc(irq);

 if (desc) {
  if (check & _IRQ_DESC_CHECK) {
   if ((check & _IRQ_DESC_PERCPU) &&
       !irq_settings_is_per_cpu_devid(desc))
    return NULL;

   if (!(check & _IRQ_DESC_PERCPU) &&
       irq_settings_is_per_cpu_devid(desc))
    return NULL;
  }

  if (bus)
   chip_bus_lock(desc);
  raw_spin_lock_irqsave(&desc->lock, *flags);
 }
 return desc;
}

void __irq_put_desc_unlock(struct irq_desc *desc, unsigned long flags, bool bus)
{
 raw_spin_unlock_irqrestore(&desc->lock, flags);
 if (bus)
  chip_bus_sync_unlock(desc);
}

int irq_set_percpu_devid_partition(unsigned int irq,
       const struct cpumask *affinity)
{
 struct irq_desc *desc = irq_to_desc(irq);

 if (!desc)
  return -EINVAL;

 if (desc->percpu_enabled)
  return -EINVAL;

 desc->percpu_enabled = kzalloc(sizeof(*desc->percpu_enabled), GFP_KERNEL);

 if (!desc->percpu_enabled)
  return -ENOMEM;

 if (affinity)
  desc->percpu_affinity = affinity;
 else
  desc->percpu_affinity = cpu_possible_mask;

 irq_set_percpu_devid_flags(irq);
 return 0;
}

int irq_set_percpu_devid(unsigned int irq)
{
 return irq_set_percpu_devid_partition(irq, NULL);
}

int irq_get_percpu_devid_partition(unsigned int irq, struct cpumask *affinity)
{
 struct irq_desc *desc = irq_to_desc(irq);

 if (!desc || !desc->percpu_enabled)
  return -EINVAL;

 if (affinity)
  cpumask_copy(affinity, desc->percpu_affinity);

 return 0;
}

void kstat_incr_irq_this_cpu(unsigned int irq)
{
 kstat_incr_irqs_this_cpu(irq_to_desc(irq));
}
unsigned int kstat_irqs_cpu(unsigned int irq, int cpu)
{
 struct irq_desc *desc = irq_to_desc(irq);

 return desc && desc->kstat_irqs ?
   *per_cpu_ptr(desc->kstat_irqs, cpu) : 0;
}
unsigned int kstat_irqs(unsigned int irq)
{
 struct irq_desc *desc = irq_to_desc(irq);
 int cpu;
 unsigned int sum = 0;

 if (!desc || !desc->kstat_irqs)
  return 0;
 for_each_possible_cpu(cpu)
  sum += *per_cpu_ptr(desc->kstat_irqs, cpu);
 return sum;
}
unsigned int kstat_irqs_usr(unsigned int irq)
{
 unsigned int sum;

 irq_lock_sparse();
 sum = kstat_irqs(irq);
 irq_unlock_sparse();
 return sum;
}


static LIST_HEAD(irq_domain_list);
static DEFINE_MUTEX(irq_domain_mutex);

static DEFINE_MUTEX(revmap_trees_mutex);
static struct irq_domain *irq_default_domain;

static void irq_domain_check_hierarchy(struct irq_domain *domain);

struct irqchip_fwid {
 struct fwnode_handle fwnode;
 char *name;
 void *data;
};
struct fwnode_handle *irq_domain_alloc_fwnode(void *data)
{
 struct irqchip_fwid *fwid;
 char *name;

 fwid = kzalloc(sizeof(*fwid), GFP_KERNEL);
 name = kasprintf(GFP_KERNEL, "irqchip@%p", data);

 if (!fwid || !name) {
  kfree(fwid);
  kfree(name);
  return NULL;
 }

 fwid->name = name;
 fwid->data = data;
 fwid->fwnode.type = FWNODE_IRQCHIP;
 return &fwid->fwnode;
}
EXPORT_SYMBOL_GPL(irq_domain_alloc_fwnode);






void irq_domain_free_fwnode(struct fwnode_handle *fwnode)
{
 struct irqchip_fwid *fwid;

 if (WARN_ON(!is_fwnode_irqchip(fwnode)))
  return;

 fwid = container_of(fwnode, struct irqchip_fwid, fwnode);
 kfree(fwid->name);
 kfree(fwid);
}
EXPORT_SYMBOL_GPL(irq_domain_free_fwnode);
struct irq_domain *__irq_domain_add(struct fwnode_handle *fwnode, int size,
        irq_hw_number_t hwirq_max, int direct_max,
        const struct irq_domain_ops *ops,
        void *host_data)
{
 struct irq_domain *domain;
 struct device_node *of_node;

 of_node = to_of_node(fwnode);

 domain = kzalloc_node(sizeof(*domain) + (sizeof(unsigned int) * size),
         GFP_KERNEL, of_node_to_nid(of_node));
 if (WARN_ON(!domain))
  return NULL;

 of_node_get(of_node);


 INIT_RADIX_TREE(&domain->revmap_tree, GFP_KERNEL);
 domain->ops = ops;
 domain->host_data = host_data;
 domain->fwnode = fwnode;
 domain->hwirq_max = hwirq_max;
 domain->revmap_size = size;
 domain->revmap_direct_max_irq = direct_max;
 irq_domain_check_hierarchy(domain);

 mutex_lock(&irq_domain_mutex);
 list_add(&domain->link, &irq_domain_list);
 mutex_unlock(&irq_domain_mutex);

 pr_debug("Added domain %s\n", domain->name);
 return domain;
}
EXPORT_SYMBOL_GPL(__irq_domain_add);
void irq_domain_remove(struct irq_domain *domain)
{
 mutex_lock(&irq_domain_mutex);

 WARN_ON(!radix_tree_empty(&domain->revmap_tree));

 list_del(&domain->link);




 if (unlikely(irq_default_domain == domain))
  irq_set_default_host(NULL);

 mutex_unlock(&irq_domain_mutex);

 pr_debug("Removed domain %s\n", domain->name);

 of_node_put(irq_domain_get_of_node(domain));
 kfree(domain);
}
EXPORT_SYMBOL_GPL(irq_domain_remove);
struct irq_domain *irq_domain_add_simple(struct device_node *of_node,
      unsigned int size,
      unsigned int first_irq,
      const struct irq_domain_ops *ops,
      void *host_data)
{
 struct irq_domain *domain;

 domain = __irq_domain_add(of_node_to_fwnode(of_node), size, size, 0, ops, host_data);
 if (!domain)
  return NULL;

 if (first_irq > 0) {
  if (IS_ENABLED(CONFIG_SPARSE_IRQ)) {

   int rc = irq_alloc_descs(first_irq, first_irq, size,
       of_node_to_nid(of_node));
   if (rc < 0)
    pr_info("Cannot allocate irq_descs @ IRQ%d, assuming pre-allocated\n",
     first_irq);
  }
  irq_domain_associate_many(domain, first_irq, 0, size);
 }

 return domain;
}
EXPORT_SYMBOL_GPL(irq_domain_add_simple);
struct irq_domain *irq_domain_add_legacy(struct device_node *of_node,
      unsigned int size,
      unsigned int first_irq,
      irq_hw_number_t first_hwirq,
      const struct irq_domain_ops *ops,
      void *host_data)
{
 struct irq_domain *domain;

 domain = __irq_domain_add(of_node_to_fwnode(of_node), first_hwirq + size,
      first_hwirq + size, 0, ops, host_data);
 if (domain)
  irq_domain_associate_many(domain, first_irq, first_hwirq, size);

 return domain;
}
EXPORT_SYMBOL_GPL(irq_domain_add_legacy);






struct irq_domain *irq_find_matching_fwspec(struct irq_fwspec *fwspec,
         enum irq_domain_bus_token bus_token)
{
 struct irq_domain *h, *found = NULL;
 struct fwnode_handle *fwnode = fwspec->fwnode;
 int rc;
 mutex_lock(&irq_domain_mutex);
 list_for_each_entry(h, &irq_domain_list, link) {
  if (h->ops->select && fwspec->param_count)
   rc = h->ops->select(h, fwspec, bus_token);
  else if (h->ops->match)
   rc = h->ops->match(h, to_of_node(fwnode), bus_token);
  else
   rc = ((fwnode != NULL) && (h->fwnode == fwnode) &&
         ((bus_token == DOMAIN_BUS_ANY) ||
          (h->bus_token == bus_token)));

  if (rc) {
   found = h;
   break;
  }
 }
 mutex_unlock(&irq_domain_mutex);
 return found;
}
EXPORT_SYMBOL_GPL(irq_find_matching_fwspec);
void irq_set_default_host(struct irq_domain *domain)
{
 pr_debug("Default domain set to @0x%p\n", domain);

 irq_default_domain = domain;
}
EXPORT_SYMBOL_GPL(irq_set_default_host);

void irq_domain_disassociate(struct irq_domain *domain, unsigned int irq)
{
 struct irq_data *irq_data = irq_get_irq_data(irq);
 irq_hw_number_t hwirq;

 if (WARN(!irq_data || irq_data->domain != domain,
   "virq%i doesn't exist; cannot disassociate\n", irq))
  return;

 hwirq = irq_data->hwirq;
 irq_set_status_flags(irq, IRQ_NOREQUEST);


 irq_set_chip_and_handler(irq, NULL, NULL);


 synchronize_irq(irq);


 if (domain->ops->unmap)
  domain->ops->unmap(domain, irq);
 smp_mb();

 irq_data->domain = NULL;
 irq_data->hwirq = 0;


 if (hwirq < domain->revmap_size) {
  domain->linear_revmap[hwirq] = 0;
 } else {
  mutex_lock(&revmap_trees_mutex);
  radix_tree_delete(&domain->revmap_tree, hwirq);
  mutex_unlock(&revmap_trees_mutex);
 }
}

int irq_domain_associate(struct irq_domain *domain, unsigned int virq,
    irq_hw_number_t hwirq)
{
 struct irq_data *irq_data = irq_get_irq_data(virq);
 int ret;

 if (WARN(hwirq >= domain->hwirq_max,
   "error: hwirq 0x%x is too large for %s\n", (int)hwirq, domain->name))
  return -EINVAL;
 if (WARN(!irq_data, "error: virq%i is not allocated", virq))
  return -EINVAL;
 if (WARN(irq_data->domain, "error: virq%i is already associated", virq))
  return -EINVAL;

 mutex_lock(&irq_domain_mutex);
 irq_data->hwirq = hwirq;
 irq_data->domain = domain;
 if (domain->ops->map) {
  ret = domain->ops->map(domain, virq, hwirq);
  if (ret != 0) {





   if (ret != -EPERM) {
    pr_info("%s didn't like hwirq-0x%lx to VIRQ%i mapping (rc=%d)\n",
           domain->name, hwirq, virq, ret);
   }
   irq_data->domain = NULL;
   irq_data->hwirq = 0;
   mutex_unlock(&irq_domain_mutex);
   return ret;
  }


  if (!domain->name && irq_data->chip)
   domain->name = irq_data->chip->name;
 }

 if (hwirq < domain->revmap_size) {
  domain->linear_revmap[hwirq] = virq;
 } else {
  mutex_lock(&revmap_trees_mutex);
  radix_tree_insert(&domain->revmap_tree, hwirq, irq_data);
  mutex_unlock(&revmap_trees_mutex);
 }
 mutex_unlock(&irq_domain_mutex);

 irq_clear_status_flags(virq, IRQ_NOREQUEST);

 return 0;
}
EXPORT_SYMBOL_GPL(irq_domain_associate);

void irq_domain_associate_many(struct irq_domain *domain, unsigned int irq_base,
          irq_hw_number_t hwirq_base, int count)
{
 struct device_node *of_node;
 int i;

 of_node = irq_domain_get_of_node(domain);
 pr_debug("%s(%s, irqbase=%i, hwbase=%i, count=%i)\n", __func__,
  of_node_full_name(of_node), irq_base, (int)hwirq_base, count);

 for (i = 0; i < count; i++) {
  irq_domain_associate(domain, irq_base + i, hwirq_base + i);
 }
}
EXPORT_SYMBOL_GPL(irq_domain_associate_many);
unsigned int irq_create_direct_mapping(struct irq_domain *domain)
{
 struct device_node *of_node;
 unsigned int virq;

 if (domain == NULL)
  domain = irq_default_domain;

 of_node = irq_domain_get_of_node(domain);
 virq = irq_alloc_desc_from(1, of_node_to_nid(of_node));
 if (!virq) {
  pr_debug("create_direct virq allocation failed\n");
  return 0;
 }
 if (virq >= domain->revmap_direct_max_irq) {
  pr_err("ERROR: no free irqs available below %i maximum\n",
   domain->revmap_direct_max_irq);
  irq_free_desc(virq);
  return 0;
 }
 pr_debug("create_direct obtained virq %d\n", virq);

 if (irq_domain_associate(domain, virq, virq)) {
  irq_free_desc(virq);
  return 0;
 }

 return virq;
}
EXPORT_SYMBOL_GPL(irq_create_direct_mapping);
unsigned int irq_create_mapping(struct irq_domain *domain,
    irq_hw_number_t hwirq)
{
 struct device_node *of_node;
 int virq;

 pr_debug("irq_create_mapping(0x%p, 0x%lx)\n", domain, hwirq);


 if (domain == NULL)
  domain = irq_default_domain;
 if (domain == NULL) {
  WARN(1, "%s(, %lx) called with NULL domain\n", __func__, hwirq);
  return 0;
 }
 pr_debug("-> using domain @%p\n", domain);

 of_node = irq_domain_get_of_node(domain);


 virq = irq_find_mapping(domain, hwirq);
 if (virq) {
  pr_debug("-> existing mapping on virq %d\n", virq);
  return virq;
 }


 virq = irq_domain_alloc_descs(-1, 1, hwirq, of_node_to_nid(of_node));
 if (virq <= 0) {
  pr_debug("-> virq allocation failed\n");
  return 0;
 }

 if (irq_domain_associate(domain, virq, hwirq)) {
  irq_free_desc(virq);
  return 0;
 }

 pr_debug("irq %lu on domain %s mapped to virtual irq %u\n",
  hwirq, of_node_full_name(of_node), virq);

 return virq;
}
EXPORT_SYMBOL_GPL(irq_create_mapping);
int irq_create_strict_mappings(struct irq_domain *domain, unsigned int irq_base,
          irq_hw_number_t hwirq_base, int count)
{
 struct device_node *of_node;
 int ret;

 of_node = irq_domain_get_of_node(domain);
 ret = irq_alloc_descs(irq_base, irq_base, count,
         of_node_to_nid(of_node));
 if (unlikely(ret < 0))
  return ret;

 irq_domain_associate_many(domain, irq_base, hwirq_base, count);
 return 0;
}
EXPORT_SYMBOL_GPL(irq_create_strict_mappings);

static int irq_domain_translate(struct irq_domain *d,
    struct irq_fwspec *fwspec,
    irq_hw_number_t *hwirq, unsigned int *type)
{
 if (d->ops->translate)
  return d->ops->translate(d, fwspec, hwirq, type);
 if (d->ops->xlate)
  return d->ops->xlate(d, to_of_node(fwspec->fwnode),
         fwspec->param, fwspec->param_count,
         hwirq, type);


 *hwirq = fwspec->param[0];
 return 0;
}

static void of_phandle_args_to_fwspec(struct of_phandle_args *irq_data,
          struct irq_fwspec *fwspec)
{
 int i;

 fwspec->fwnode = irq_data->np ? &irq_data->np->fwnode : NULL;
 fwspec->param_count = irq_data->args_count;

 for (i = 0; i < irq_data->args_count; i++)
  fwspec->param[i] = irq_data->args[i];
}

unsigned int irq_create_fwspec_mapping(struct irq_fwspec *fwspec)
{
 struct irq_domain *domain;
 irq_hw_number_t hwirq;
 unsigned int type = IRQ_TYPE_NONE;
 int virq;

 if (fwspec->fwnode) {
  domain = irq_find_matching_fwspec(fwspec, DOMAIN_BUS_WIRED);
  if (!domain)
   domain = irq_find_matching_fwspec(fwspec, DOMAIN_BUS_ANY);
 } else {
  domain = irq_default_domain;
 }

 if (!domain) {
  pr_warn("no irq domain found for %s !\n",
   of_node_full_name(to_of_node(fwspec->fwnode)));
  return 0;
 }

 if (irq_domain_translate(domain, fwspec, &hwirq, &type))
  return 0;

 if (irq_domain_is_hierarchy(domain)) {




  virq = irq_find_mapping(domain, hwirq);
  if (virq)
   return virq;

  virq = irq_domain_alloc_irqs(domain, 1, NUMA_NO_NODE, fwspec);
  if (virq <= 0)
   return 0;
 } else {

  virq = irq_create_mapping(domain, hwirq);
  if (!virq)
   return virq;
 }


 if (type != IRQ_TYPE_NONE &&
     type != irq_get_trigger_type(virq))
  irq_set_irq_type(virq, type);
 return virq;
}
EXPORT_SYMBOL_GPL(irq_create_fwspec_mapping);

unsigned int irq_create_of_mapping(struct of_phandle_args *irq_data)
{
 struct irq_fwspec fwspec;

 of_phandle_args_to_fwspec(irq_data, &fwspec);
 return irq_create_fwspec_mapping(&fwspec);
}
EXPORT_SYMBOL_GPL(irq_create_of_mapping);





void irq_dispose_mapping(unsigned int virq)
{
 struct irq_data *irq_data = irq_get_irq_data(virq);
 struct irq_domain *domain;

 if (!virq || !irq_data)
  return;

 domain = irq_data->domain;
 if (WARN_ON(domain == NULL))
  return;

 irq_domain_disassociate(domain, virq);
 irq_free_desc(virq);
}
EXPORT_SYMBOL_GPL(irq_dispose_mapping);






unsigned int irq_find_mapping(struct irq_domain *domain,
         irq_hw_number_t hwirq)
{
 struct irq_data *data;


 if (domain == NULL)
  domain = irq_default_domain;
 if (domain == NULL)
  return 0;

 if (hwirq < domain->revmap_direct_max_irq) {
  data = irq_domain_get_irq_data(domain, hwirq);
  if (data && data->hwirq == hwirq)
   return hwirq;
 }


 if (hwirq < domain->revmap_size)
  return domain->linear_revmap[hwirq];

 rcu_read_lock();
 data = radix_tree_lookup(&domain->revmap_tree, hwirq);
 rcu_read_unlock();
 return data ? data->irq : 0;
}
EXPORT_SYMBOL_GPL(irq_find_mapping);

static int virq_debug_show(struct seq_file *m, void *private)
{
 unsigned long flags;
 struct irq_desc *desc;
 struct irq_domain *domain;
 struct radix_tree_iter iter;
 void *data, **slot;
 int i;

 seq_printf(m, " %-16s  %-6s  %-10s  %-10s  %s\n",
     "name", "mapped", "linear-max", "direct-max", "devtree-node");
 mutex_lock(&irq_domain_mutex);
 list_for_each_entry(domain, &irq_domain_list, link) {
  struct device_node *of_node;
  int count = 0;
  of_node = irq_domain_get_of_node(domain);
  radix_tree_for_each_slot(slot, &domain->revmap_tree, &iter, 0)
   count++;
  seq_printf(m, "%c%-16s  %6u  %10u  %10u  %s\n",
      domain == irq_default_domain ? '*' : ' ', domain->name,
      domain->revmap_size + count, domain->revmap_size,
      domain->revmap_direct_max_irq,
      of_node ? of_node_full_name(of_node) : "");
 }
 mutex_unlock(&irq_domain_mutex);

 seq_printf(m, "%-5s  %-7s  %-15s  %-*s  %6s  %-14s  %s\n", "irq", "hwirq",
        "chip name", (int)(2 * sizeof(void *) + 2), "chip data",
        "active", "type", "domain");

 for (i = 1; i < nr_irqs; i++) {
  desc = irq_to_desc(i);
  if (!desc)
   continue;

  raw_spin_lock_irqsave(&desc->lock, flags);
  domain = desc->irq_data.domain;

  if (domain) {
   struct irq_chip *chip;
   int hwirq = desc->irq_data.hwirq;
   bool direct;

   seq_printf(m, "%5d  ", i);
   seq_printf(m, "0x%05x  ", hwirq);

   chip = irq_desc_get_chip(desc);
   seq_printf(m, "%-15s  ", (chip && chip->name) ? chip->name : "none");

   data = irq_desc_get_chip_data(desc);
   seq_printf(m, data ? "0x%p  " : "  %p  ", data);

   seq_printf(m, "   %c    ", (desc->action && desc->action->handler) ? '*' : ' ');
   direct = (i == hwirq) && (i < domain->revmap_direct_max_irq);
   seq_printf(m, "%6s%-8s  ",
       (hwirq < domain->revmap_size) ? "LINEAR" : "RADIX",
       direct ? "(DIRECT)" : "");
   seq_printf(m, "%s\n", desc->irq_data.domain->name);
  }

  raw_spin_unlock_irqrestore(&desc->lock, flags);
 }

 return 0;
}

static int virq_debug_open(struct inode *inode, struct file *file)
{
 return single_open(file, virq_debug_show, inode->i_private);
}

static const struct file_operations virq_debug_fops = {
 .open = virq_debug_open,
 .read = seq_read,
 .llseek = seq_lseek,
 .release = single_release,
};

static int __init irq_debugfs_init(void)
{
 if (debugfs_create_file("irq_domain_mapping", S_IRUGO, NULL,
     NULL, &virq_debug_fops) == NULL)
  return -ENOMEM;

 return 0;
}
__initcall(irq_debugfs_init);







int irq_domain_xlate_onecell(struct irq_domain *d, struct device_node *ctrlr,
        const u32 *intspec, unsigned int intsize,
        unsigned long *out_hwirq, unsigned int *out_type)
{
 if (WARN_ON(intsize < 1))
  return -EINVAL;
 *out_hwirq = intspec[0];
 *out_type = IRQ_TYPE_NONE;
 return 0;
}
EXPORT_SYMBOL_GPL(irq_domain_xlate_onecell);
int irq_domain_xlate_twocell(struct irq_domain *d, struct device_node *ctrlr,
   const u32 *intspec, unsigned int intsize,
   irq_hw_number_t *out_hwirq, unsigned int *out_type)
{
 if (WARN_ON(intsize < 2))
  return -EINVAL;
 *out_hwirq = intspec[0];
 *out_type = intspec[1] & IRQ_TYPE_SENSE_MASK;
 return 0;
}
EXPORT_SYMBOL_GPL(irq_domain_xlate_twocell);
int irq_domain_xlate_onetwocell(struct irq_domain *d,
    struct device_node *ctrlr,
    const u32 *intspec, unsigned int intsize,
    unsigned long *out_hwirq, unsigned int *out_type)
{
 if (WARN_ON(intsize < 1))
  return -EINVAL;
 *out_hwirq = intspec[0];
 *out_type = (intsize > 1) ? intspec[1] : IRQ_TYPE_NONE;
 return 0;
}
EXPORT_SYMBOL_GPL(irq_domain_xlate_onetwocell);

const struct irq_domain_ops irq_domain_simple_ops = {
 .xlate = irq_domain_xlate_onetwocell,
};
EXPORT_SYMBOL_GPL(irq_domain_simple_ops);

int irq_domain_alloc_descs(int virq, unsigned int cnt, irq_hw_number_t hwirq,
      int node)
{
 unsigned int hint;

 if (virq >= 0) {
  virq = irq_alloc_descs(virq, virq, cnt, node);
 } else {
  hint = hwirq % nr_irqs;
  if (hint == 0)
   hint++;
  virq = irq_alloc_descs_from(hint, cnt, node);
  if (virq <= 0 && hint > 1)
   virq = irq_alloc_descs_from(1, cnt, node);
 }

 return virq;
}

struct irq_domain *irq_domain_create_hierarchy(struct irq_domain *parent,
         unsigned int flags,
         unsigned int size,
         struct fwnode_handle *fwnode,
         const struct irq_domain_ops *ops,
         void *host_data)
{
 struct irq_domain *domain;

 if (size)
  domain = irq_domain_create_linear(fwnode, size, ops, host_data);
 else
  domain = irq_domain_create_tree(fwnode, ops, host_data);
 if (domain) {
  domain->parent = parent;
  domain->flags |= flags;
 }

 return domain;
}
EXPORT_SYMBOL_GPL(irq_domain_create_hierarchy);

static void irq_domain_insert_irq(int virq)
{
 struct irq_data *data;

 for (data = irq_get_irq_data(virq); data; data = data->parent_data) {
  struct irq_domain *domain = data->domain;
  irq_hw_number_t hwirq = data->hwirq;

  if (hwirq < domain->revmap_size) {
   domain->linear_revmap[hwirq] = virq;
  } else {
   mutex_lock(&revmap_trees_mutex);
   radix_tree_insert(&domain->revmap_tree, hwirq, data);
   mutex_unlock(&revmap_trees_mutex);
  }


  if (!domain->name && data->chip)
   domain->name = data->chip->name;
 }

 irq_clear_status_flags(virq, IRQ_NOREQUEST);
}

static void irq_domain_remove_irq(int virq)
{
 struct irq_data *data;

 irq_set_status_flags(virq, IRQ_NOREQUEST);
 irq_set_chip_and_handler(virq, NULL, NULL);
 synchronize_irq(virq);
 smp_mb();

 for (data = irq_get_irq_data(virq); data; data = data->parent_data) {
  struct irq_domain *domain = data->domain;
  irq_hw_number_t hwirq = data->hwirq;

  if (hwirq < domain->revmap_size) {
   domain->linear_revmap[hwirq] = 0;
  } else {
   mutex_lock(&revmap_trees_mutex);
   radix_tree_delete(&domain->revmap_tree, hwirq);
   mutex_unlock(&revmap_trees_mutex);
  }
 }
}

static struct irq_data *irq_domain_insert_irq_data(struct irq_domain *domain,
         struct irq_data *child)
{
 struct irq_data *irq_data;

 irq_data = kzalloc_node(sizeof(*irq_data), GFP_KERNEL,
    irq_data_get_node(child));
 if (irq_data) {
  child->parent_data = irq_data;
  irq_data->irq = child->irq;
  irq_data->common = child->common;
  irq_data->domain = domain;
 }

 return irq_data;
}

static void irq_domain_free_irq_data(unsigned int virq, unsigned int nr_irqs)
{
 struct irq_data *irq_data, *tmp;
 int i;

 for (i = 0; i < nr_irqs; i++) {
  irq_data = irq_get_irq_data(virq + i);
  tmp = irq_data->parent_data;
  irq_data->parent_data = NULL;
  irq_data->domain = NULL;

  while (tmp) {
   irq_data = tmp;
   tmp = tmp->parent_data;
   kfree(irq_data);
  }
 }
}

static int irq_domain_alloc_irq_data(struct irq_domain *domain,
         unsigned int virq, unsigned int nr_irqs)
{
 struct irq_data *irq_data;
 struct irq_domain *parent;
 int i;


 for (i = 0; i < nr_irqs; i++) {
  irq_data = irq_get_irq_data(virq + i);
  irq_data->domain = domain;

  for (parent = domain->parent; parent; parent = parent->parent) {
   irq_data = irq_domain_insert_irq_data(parent, irq_data);
   if (!irq_data) {
    irq_domain_free_irq_data(virq, i + 1);
    return -ENOMEM;
   }
  }
 }

 return 0;
}






struct irq_data *irq_domain_get_irq_data(struct irq_domain *domain,
      unsigned int virq)
{
 struct irq_data *irq_data;

 for (irq_data = irq_get_irq_data(virq); irq_data;
      irq_data = irq_data->parent_data)
  if (irq_data->domain == domain)
   return irq_data;

 return NULL;
}
EXPORT_SYMBOL_GPL(irq_domain_get_irq_data);
int irq_domain_set_hwirq_and_chip(struct irq_domain *domain, unsigned int virq,
      irq_hw_number_t hwirq, struct irq_chip *chip,
      void *chip_data)
{
 struct irq_data *irq_data = irq_domain_get_irq_data(domain, virq);

 if (!irq_data)
  return -ENOENT;

 irq_data->hwirq = hwirq;
 irq_data->chip = chip ? chip : &no_irq_chip;
 irq_data->chip_data = chip_data;

 return 0;
}
EXPORT_SYMBOL_GPL(irq_domain_set_hwirq_and_chip);
void irq_domain_set_info(struct irq_domain *domain, unsigned int virq,
    irq_hw_number_t hwirq, struct irq_chip *chip,
    void *chip_data, irq_flow_handler_t handler,
    void *handler_data, const char *handler_name)
{
 irq_domain_set_hwirq_and_chip(domain, virq, hwirq, chip, chip_data);
 __irq_set_handler(virq, handler, 0, handler_name);
 irq_set_handler_data(virq, handler_data);
}
EXPORT_SYMBOL(irq_domain_set_info);





void irq_domain_reset_irq_data(struct irq_data *irq_data)
{
 irq_data->hwirq = 0;
 irq_data->chip = &no_irq_chip;
 irq_data->chip_data = NULL;
}
EXPORT_SYMBOL_GPL(irq_domain_reset_irq_data);







void irq_domain_free_irqs_common(struct irq_domain *domain, unsigned int virq,
     unsigned int nr_irqs)
{
 struct irq_data *irq_data;
 int i;

 for (i = 0; i < nr_irqs; i++) {
  irq_data = irq_domain_get_irq_data(domain, virq + i);
  if (irq_data)
   irq_domain_reset_irq_data(irq_data);
 }
 irq_domain_free_irqs_parent(domain, virq, nr_irqs);
}
EXPORT_SYMBOL_GPL(irq_domain_free_irqs_common);







void irq_domain_free_irqs_top(struct irq_domain *domain, unsigned int virq,
         unsigned int nr_irqs)
{
 int i;

 for (i = 0; i < nr_irqs; i++) {
  irq_set_handler_data(virq + i, NULL);
  irq_set_handler(virq + i, NULL);
 }
 irq_domain_free_irqs_common(domain, virq, nr_irqs);
}

static bool irq_domain_is_auto_recursive(struct irq_domain *domain)
{
 return domain->flags & IRQ_DOMAIN_FLAG_AUTO_RECURSIVE;
}

static void irq_domain_free_irqs_recursive(struct irq_domain *domain,
        unsigned int irq_base,
        unsigned int nr_irqs)
{
 domain->ops->free(domain, irq_base, nr_irqs);
 if (irq_domain_is_auto_recursive(domain)) {
  BUG_ON(!domain->parent);
  irq_domain_free_irqs_recursive(domain->parent, irq_base,
            nr_irqs);
 }
}

int irq_domain_alloc_irqs_recursive(struct irq_domain *domain,
        unsigned int irq_base,
        unsigned int nr_irqs, void *arg)
{
 int ret = 0;
 struct irq_domain *parent = domain->parent;
 bool recursive = irq_domain_is_auto_recursive(domain);

 BUG_ON(recursive && !parent);
 if (recursive)
  ret = irq_domain_alloc_irqs_recursive(parent, irq_base,
            nr_irqs, arg);
 if (ret >= 0)
  ret = domain->ops->alloc(domain, irq_base, nr_irqs, arg);
 if (ret < 0 && recursive)
  irq_domain_free_irqs_recursive(parent, irq_base, nr_irqs);

 return ret;
}
int __irq_domain_alloc_irqs(struct irq_domain *domain, int irq_base,
       unsigned int nr_irqs, int node, void *arg,
       bool realloc)
{
 int i, ret, virq;

 if (domain == NULL) {
  domain = irq_default_domain;
  if (WARN(!domain, "domain is NULL; cannot allocate IRQ\n"))
   return -EINVAL;
 }

 if (!domain->ops->alloc) {
  pr_debug("domain->ops->alloc() is NULL\n");
  return -ENOSYS;
 }

 if (realloc && irq_base >= 0) {
  virq = irq_base;
 } else {
  virq = irq_domain_alloc_descs(irq_base, nr_irqs, 0, node);
  if (virq < 0) {
   pr_debug("cannot allocate IRQ(base %d, count %d)\n",
     irq_base, nr_irqs);
   return virq;
  }
 }

 if (irq_domain_alloc_irq_data(domain, virq, nr_irqs)) {
  pr_debug("cannot allocate memory for IRQ%d\n", virq);
  ret = -ENOMEM;
  goto out_free_desc;
 }

 mutex_lock(&irq_domain_mutex);
 ret = irq_domain_alloc_irqs_recursive(domain, virq, nr_irqs, arg);
 if (ret < 0) {
  mutex_unlock(&irq_domain_mutex);
  goto out_free_irq_data;
 }
 for (i = 0; i < nr_irqs; i++)
  irq_domain_insert_irq(virq + i);
 mutex_unlock(&irq_domain_mutex);

 return virq;

out_free_irq_data:
 irq_domain_free_irq_data(virq, nr_irqs);
out_free_desc:
 irq_free_descs(virq, nr_irqs);
 return ret;
}






void irq_domain_free_irqs(unsigned int virq, unsigned int nr_irqs)
{
 struct irq_data *data = irq_get_irq_data(virq);
 int i;

 if (WARN(!data || !data->domain || !data->domain->ops->free,
   "NULL pointer, cannot free irq\n"))
  return;

 mutex_lock(&irq_domain_mutex);
 for (i = 0; i < nr_irqs; i++)
  irq_domain_remove_irq(virq + i);
 irq_domain_free_irqs_recursive(data->domain, virq, nr_irqs);
 mutex_unlock(&irq_domain_mutex);

 irq_domain_free_irq_data(virq, nr_irqs);
 irq_free_descs(virq, nr_irqs);
}
int irq_domain_alloc_irqs_parent(struct irq_domain *domain,
     unsigned int irq_base, unsigned int nr_irqs,
     void *arg)
{

 if (irq_domain_is_auto_recursive(domain))
  return 0;

 domain = domain->parent;
 if (domain)
  return irq_domain_alloc_irqs_recursive(domain, irq_base,
             nr_irqs, arg);
 return -ENOSYS;
}
EXPORT_SYMBOL_GPL(irq_domain_alloc_irqs_parent);
void irq_domain_free_irqs_parent(struct irq_domain *domain,
     unsigned int irq_base, unsigned int nr_irqs)
{

 if (!irq_domain_is_auto_recursive(domain) && domain->parent)
  irq_domain_free_irqs_recursive(domain->parent, irq_base,
            nr_irqs);
}
EXPORT_SYMBOL_GPL(irq_domain_free_irqs_parent);
void irq_domain_activate_irq(struct irq_data *irq_data)
{
 if (irq_data && irq_data->domain) {
  struct irq_domain *domain = irq_data->domain;

  if (irq_data->parent_data)
   irq_domain_activate_irq(irq_data->parent_data);
  if (domain->ops->activate)
   domain->ops->activate(domain, irq_data);
 }
}
void irq_domain_deactivate_irq(struct irq_data *irq_data)
{
 if (irq_data && irq_data->domain) {
  struct irq_domain *domain = irq_data->domain;

  if (domain->ops->deactivate)
   domain->ops->deactivate(domain, irq_data);
  if (irq_data->parent_data)
   irq_domain_deactivate_irq(irq_data->parent_data);
 }
}

static void irq_domain_check_hierarchy(struct irq_domain *domain)
{

 if (domain->ops->alloc)
  domain->flags |= IRQ_DOMAIN_FLAG_HIERARCHY;
}





struct irq_data *irq_domain_get_irq_data(struct irq_domain *domain,
      unsigned int virq)
{
 struct irq_data *irq_data = irq_get_irq_data(virq);

 return (irq_data && irq_data->domain == domain) ? irq_data : NULL;
}
EXPORT_SYMBOL_GPL(irq_domain_get_irq_data);
void irq_domain_set_info(struct irq_domain *domain, unsigned int virq,
    irq_hw_number_t hwirq, struct irq_chip *chip,
    void *chip_data, irq_flow_handler_t handler,
    void *handler_data, const char *handler_name)
{
 irq_set_chip_and_handler_name(virq, chip, handler, handler_name);
 irq_set_chip_data(virq, chip_data);
 irq_set_handler_data(virq, handler_data);
}

static void irq_domain_check_hierarchy(struct irq_domain *domain)
{
}









static DEFINE_PER_CPU(struct llist_head, raised_list);
static DEFINE_PER_CPU(struct llist_head, lazy_list);




static bool irq_work_claim(struct irq_work *work)
{
 unsigned long flags, oflags, nflags;





 flags = work->flags & ~IRQ_WORK_PENDING;
 for (;;) {
  nflags = flags | IRQ_WORK_FLAGS;
  oflags = cmpxchg(&work->flags, flags, nflags);
  if (oflags == flags)
   break;
  if (oflags & IRQ_WORK_PENDING)
   return false;
  flags = oflags;
  cpu_relax();
 }

 return true;
}

void __weak arch_irq_work_raise(void)
{



}







bool irq_work_queue_on(struct irq_work *work, int cpu)
{

 WARN_ON_ONCE(cpu_is_offline(cpu));


 WARN_ON_ONCE(in_nmi());


 if (!irq_work_claim(work))
  return false;

 if (llist_add(&work->llnode, &per_cpu(raised_list, cpu)))
  arch_send_call_function_single_ipi(cpu);

 return true;
}
EXPORT_SYMBOL_GPL(irq_work_queue_on);


bool irq_work_queue(struct irq_work *work)
{

 if (!irq_work_claim(work))
  return false;


 preempt_disable();


 if (work->flags & IRQ_WORK_LAZY) {
  if (llist_add(&work->llnode, this_cpu_ptr(&lazy_list)) &&
      tick_nohz_tick_stopped())
   arch_irq_work_raise();
 } else {
  if (llist_add(&work->llnode, this_cpu_ptr(&raised_list)))
   arch_irq_work_raise();
 }

 preempt_enable();

 return true;
}
EXPORT_SYMBOL_GPL(irq_work_queue);

bool irq_work_needs_cpu(void)
{
 struct llist_head *raised, *lazy;

 raised = this_cpu_ptr(&raised_list);
 lazy = this_cpu_ptr(&lazy_list);

 if (llist_empty(raised) || arch_irq_work_has_interrupt())
  if (llist_empty(lazy))
   return false;


 WARN_ON_ONCE(cpu_is_offline(smp_processor_id()));

 return true;
}

static void irq_work_run_list(struct llist_head *list)
{
 unsigned long flags;
 struct irq_work *work;
 struct llist_node *llnode;

 BUG_ON(!irqs_disabled());

 if (llist_empty(list))
  return;

 llnode = llist_del_all(list);
 while (llnode != NULL) {
  work = llist_entry(llnode, struct irq_work, llnode);

  llnode = llist_next(llnode);
  flags = work->flags & ~IRQ_WORK_PENDING;
  xchg(&work->flags, flags);

  work->func(work);




  (void)cmpxchg(&work->flags, flags, flags & ~IRQ_WORK_BUSY);
 }
}





void irq_work_run(void)
{
 irq_work_run_list(this_cpu_ptr(&raised_list));
 irq_work_run_list(this_cpu_ptr(&lazy_list));
}
EXPORT_SYMBOL_GPL(irq_work_run);

void irq_work_tick(void)
{
 struct llist_head *raised = this_cpu_ptr(&raised_list);

 if (!llist_empty(raised) && !arch_irq_work_has_interrupt())
  irq_work_run_list(raised);
 irq_work_run_list(this_cpu_ptr(&lazy_list));
}





void irq_work_sync(struct irq_work *work)
{
 WARN_ON_ONCE(irqs_disabled());

 while (work->flags & IRQ_WORK_BUSY)
  cpu_relax();
}
EXPORT_SYMBOL_GPL(irq_work_sync);

static struct timeval itimer_get_remtime(struct hrtimer *timer)
{
 ktime_t rem = __hrtimer_get_remaining(timer, true);






 if (hrtimer_active(timer)) {
  if (rem.tv64 <= 0)
   rem.tv64 = NSEC_PER_USEC;
 } else
  rem.tv64 = 0;

 return ktime_to_timeval(rem);
}

static void get_cpu_itimer(struct task_struct *tsk, unsigned int clock_id,
      struct itimerval *const value)
{
 cputime_t cval, cinterval;
 struct cpu_itimer *it = &tsk->signal->it[clock_id];

 spin_lock_irq(&tsk->sighand->siglock);

 cval = it->expires;
 cinterval = it->incr;
 if (cval) {
  struct task_cputime cputime;
  cputime_t t;

  thread_group_cputimer(tsk, &cputime);
  if (clock_id == CPUCLOCK_PROF)
   t = cputime.utime + cputime.stime;
  else

   t = cputime.utime;

  if (cval < t)

   cval = cputime_one_jiffy;
  else
   cval = cval - t;
 }

 spin_unlock_irq(&tsk->sighand->siglock);

 cputime_to_timeval(cval, &value->it_value);
 cputime_to_timeval(cinterval, &value->it_interval);
}

int do_getitimer(int which, struct itimerval *value)
{
 struct task_struct *tsk = current;

 switch (which) {
 case ITIMER_REAL:
  spin_lock_irq(&tsk->sighand->siglock);
  value->it_value = itimer_get_remtime(&tsk->signal->real_timer);
  value->it_interval =
   ktime_to_timeval(tsk->signal->it_real_incr);
  spin_unlock_irq(&tsk->sighand->siglock);
  break;
 case ITIMER_VIRTUAL:
  get_cpu_itimer(tsk, CPUCLOCK_VIRT, value);
  break;
 case ITIMER_PROF:
  get_cpu_itimer(tsk, CPUCLOCK_PROF, value);
  break;
 default:
  return(-EINVAL);
 }
 return 0;
}

SYSCALL_DEFINE2(getitimer, int, which, struct itimerval __user *, value)
{
 int error = -EFAULT;
 struct itimerval get_buffer;

 if (value) {
  error = do_getitimer(which, &get_buffer);
  if (!error &&
      copy_to_user(value, &get_buffer, sizeof(get_buffer)))
   error = -EFAULT;
 }
 return error;
}





enum hrtimer_restart it_real_fn(struct hrtimer *timer)
{
 struct signal_struct *sig =
  container_of(timer, struct signal_struct, real_timer);

 trace_itimer_expire(ITIMER_REAL, sig->leader_pid, 0);
 kill_pid_info(SIGALRM, SEND_SIG_PRIV, sig->leader_pid);

 return HRTIMER_NORESTART;
}

static inline u32 cputime_sub_ns(cputime_t ct, s64 real_ns)
{
 struct timespec ts;
 s64 cpu_ns;

 cputime_to_timespec(ct, &ts);
 cpu_ns = timespec_to_ns(&ts);

 return (cpu_ns <= real_ns) ? 0 : cpu_ns - real_ns;
}

static void set_cpu_itimer(struct task_struct *tsk, unsigned int clock_id,
      const struct itimerval *const value,
      struct itimerval *const ovalue)
{
 cputime_t cval, nval, cinterval, ninterval;
 s64 ns_ninterval, ns_nval;
 u32 error, incr_error;
 struct cpu_itimer *it = &tsk->signal->it[clock_id];

 nval = timeval_to_cputime(&value->it_value);
 ns_nval = timeval_to_ns(&value->it_value);
 ninterval = timeval_to_cputime(&value->it_interval);
 ns_ninterval = timeval_to_ns(&value->it_interval);

 error = cputime_sub_ns(nval, ns_nval);
 incr_error = cputime_sub_ns(ninterval, ns_ninterval);

 spin_lock_irq(&tsk->sighand->siglock);

 cval = it->expires;
 cinterval = it->incr;
 if (cval || nval) {
  if (nval > 0)
   nval += cputime_one_jiffy;
  set_process_cpu_timer(tsk, clock_id, &nval, &cval);
 }
 it->expires = nval;
 it->incr = ninterval;
 it->error = error;
 it->incr_error = incr_error;
 trace_itimer_state(clock_id == CPUCLOCK_VIRT ?
      ITIMER_VIRTUAL : ITIMER_PROF, value, nval);

 spin_unlock_irq(&tsk->sighand->siglock);

 if (ovalue) {
  cputime_to_timeval(cval, &ovalue->it_value);
  cputime_to_timeval(cinterval, &ovalue->it_interval);
 }
}




 (((t)->tv_sec >= 0) && (((unsigned long) (t)->tv_usec) < USEC_PER_SEC))

int do_setitimer(int which, struct itimerval *value, struct itimerval *ovalue)
{
 struct task_struct *tsk = current;
 struct hrtimer *timer;
 ktime_t expires;




 if (!timeval_valid(&value->it_value) ||
     !timeval_valid(&value->it_interval))
  return -EINVAL;

 switch (which) {
 case ITIMER_REAL:
again:
  spin_lock_irq(&tsk->sighand->siglock);
  timer = &tsk->signal->real_timer;
  if (ovalue) {
   ovalue->it_value = itimer_get_remtime(timer);
   ovalue->it_interval
    = ktime_to_timeval(tsk->signal->it_real_incr);
  }

  if (hrtimer_try_to_cancel(timer) < 0) {
   spin_unlock_irq(&tsk->sighand->siglock);
   goto again;
  }
  expires = timeval_to_ktime(value->it_value);
  if (expires.tv64 != 0) {
   tsk->signal->it_real_incr =
    timeval_to_ktime(value->it_interval);
   hrtimer_start(timer, expires, HRTIMER_MODE_REL);
  } else
   tsk->signal->it_real_incr.tv64 = 0;

  trace_itimer_state(ITIMER_REAL, value, 0);
  spin_unlock_irq(&tsk->sighand->siglock);
  break;
 case ITIMER_VIRTUAL:
  set_cpu_itimer(tsk, CPUCLOCK_VIRT, value, ovalue);
  break;
 case ITIMER_PROF:
  set_cpu_itimer(tsk, CPUCLOCK_PROF, value, ovalue);
  break;
 default:
  return -EINVAL;
 }
 return 0;
}
unsigned int alarm_setitimer(unsigned int seconds)
{
 struct itimerval it_new, it_old;

 if (seconds > INT_MAX)
  seconds = INT_MAX;
 it_new.it_value.tv_sec = seconds;
 it_new.it_value.tv_usec = 0;
 it_new.it_interval.tv_sec = it_new.it_interval.tv_usec = 0;

 do_setitimer(ITIMER_REAL, &it_new, &it_old);





 if ((!it_old.it_value.tv_sec && it_old.it_value.tv_usec) ||
       it_old.it_value.tv_usec >= 500000)
  it_old.it_value.tv_sec++;

 return it_old.it_value.tv_sec;
}

SYSCALL_DEFINE3(setitimer, int, which, struct itimerval __user *, value,
  struct itimerval __user *, ovalue)
{
 struct itimerval set_buffer, get_buffer;
 int error;

 if (value) {
  if(copy_from_user(&set_buffer, value, sizeof(set_buffer)))
   return -EFAULT;
 } else {
  memset(&set_buffer, 0, sizeof(set_buffer));
  printk_once(KERN_WARNING "%s calls setitimer() with new_value NULL pointer."
       " Misfeature support will be removed\n",
       current->comm);
 }

 error = do_setitimer(which, &set_buffer, ovalue ? &get_buffer : NULL);
 if (error || !ovalue)
  return error;

 if (copy_to_user(ovalue, &get_buffer, sizeof(get_buffer)))
  return -EFAULT;
 return 0;
}


static cycle_t jiffies_read(struct clocksource *cs)
{
 return (cycle_t) jiffies;
}

static struct clocksource clocksource_jiffies = {
 .name = "jiffies",
 .rating = 1,
 .read = jiffies_read,
 .mask = CLOCKSOURCE_MASK(32),
 .mult = NSEC_PER_JIFFY << JIFFIES_SHIFT,
 .shift = JIFFIES_SHIFT,
 .max_cycles = 10,
};

__cacheline_aligned_in_smp DEFINE_SEQLOCK(jiffies_lock);

u64 get_jiffies_64(void)
{
 unsigned long seq;
 u64 ret;

 do {
  seq = read_seqbegin(&jiffies_lock);
  ret = jiffies_64;
 } while (read_seqretry(&jiffies_lock, seq));
 return ret;
}
EXPORT_SYMBOL(get_jiffies_64);

EXPORT_SYMBOL(jiffies);

static int __init init_jiffies_clocksource(void)
{
 return __clocksource_register(&clocksource_jiffies);
}

core_initcall(init_jiffies_clocksource);

struct clocksource * __init __weak clocksource_default_clock(void)
{
 return &clocksource_jiffies;
}

struct clocksource refined_jiffies;

int register_refined_jiffies(long cycles_per_second)
{
 u64 nsec_per_tick, shift_hz;
 long cycles_per_tick;



 refined_jiffies = clocksource_jiffies;
 refined_jiffies.name = "refined-jiffies";
 refined_jiffies.rating++;


 cycles_per_tick = (cycles_per_second + HZ/2)/HZ;

 shift_hz = (u64)cycles_per_second << 8;
 shift_hz += cycles_per_tick/2;
 do_div(shift_hz, cycles_per_tick);

 nsec_per_tick = (u64)NSEC_PER_SEC << 8;
 nsec_per_tick += (u32)shift_hz/2;
 do_div(nsec_per_tick, (u32)shift_hz);

 refined_jiffies.mult = ((u32)nsec_per_tick) << JIFFIES_SHIFT;

 __clocksource_register(&refined_jiffies);
 return 0;
}










static DEFINE_MUTEX(jump_label_mutex);

void jump_label_lock(void)
{
 mutex_lock(&jump_label_mutex);
}

void jump_label_unlock(void)
{
 mutex_unlock(&jump_label_mutex);
}

static int jump_label_cmp(const void *a, const void *b)
{
 const struct jump_entry *jea = a;
 const struct jump_entry *jeb = b;

 if (jea->key < jeb->key)
  return -1;

 if (jea->key > jeb->key)
  return 1;

 return 0;
}

static void
jump_label_sort_entries(struct jump_entry *start, struct jump_entry *stop)
{
 unsigned long size;

 size = (((unsigned long)stop - (unsigned long)start)
     / sizeof(struct jump_entry));
 sort(start, size, sizeof(struct jump_entry), jump_label_cmp, NULL);
}

static void jump_label_update(struct static_key *key);

void static_key_slow_inc(struct static_key *key)
{
 int v, v1;

 STATIC_KEY_CHECK_USE();
 for (v = atomic_read(&key->enabled); v > 0; v = v1) {
  v1 = atomic_cmpxchg(&key->enabled, v, v + 1);
  if (likely(v1 == v))
   return;
 }

 jump_label_lock();
 if (atomic_read(&key->enabled) == 0) {
  atomic_set(&key->enabled, -1);
  jump_label_update(key);
  atomic_set(&key->enabled, 1);
 } else {
  atomic_inc(&key->enabled);
 }
 jump_label_unlock();
}
EXPORT_SYMBOL_GPL(static_key_slow_inc);

static void __static_key_slow_dec(struct static_key *key,
  unsigned long rate_limit, struct delayed_work *work)
{







 if (!atomic_dec_and_mutex_lock(&key->enabled, &jump_label_mutex)) {
  WARN(atomic_read(&key->enabled) < 0,
       "jump label: negative count!\n");
  return;
 }

 if (rate_limit) {
  atomic_inc(&key->enabled);
  schedule_delayed_work(work, rate_limit);
 } else {
  jump_label_update(key);
 }
 jump_label_unlock();
}

static void jump_label_update_timeout(struct work_struct *work)
{
 struct static_key_deferred *key =
  container_of(work, struct static_key_deferred, work.work);
 __static_key_slow_dec(&key->key, 0, NULL);
}

void static_key_slow_dec(struct static_key *key)
{
 STATIC_KEY_CHECK_USE();
 __static_key_slow_dec(key, 0, NULL);
}
EXPORT_SYMBOL_GPL(static_key_slow_dec);

void static_key_slow_dec_deferred(struct static_key_deferred *key)
{
 STATIC_KEY_CHECK_USE();
 __static_key_slow_dec(&key->key, key->timeout, &key->work);
}
EXPORT_SYMBOL_GPL(static_key_slow_dec_deferred);

void jump_label_rate_limit(struct static_key_deferred *key,
  unsigned long rl)
{
 STATIC_KEY_CHECK_USE();
 key->timeout = rl;
 INIT_DELAYED_WORK(&key->work, jump_label_update_timeout);
}
EXPORT_SYMBOL_GPL(jump_label_rate_limit);

static int addr_conflict(struct jump_entry *entry, void *start, void *end)
{
 if (entry->code <= (unsigned long)end &&
  entry->code + JUMP_LABEL_NOP_SIZE > (unsigned long)start)
  return 1;

 return 0;
}

static int __jump_label_text_reserved(struct jump_entry *iter_start,
  struct jump_entry *iter_stop, void *start, void *end)
{
 struct jump_entry *iter;

 iter = iter_start;
 while (iter < iter_stop) {
  if (addr_conflict(iter, start, end))
   return 1;
  iter++;
 }

 return 0;
}







void __weak __init_or_module arch_jump_label_transform_static(struct jump_entry *entry,
         enum jump_label_type type)
{
 arch_jump_label_transform(entry, type);
}

static inline struct jump_entry *static_key_entries(struct static_key *key)
{
 return (struct jump_entry *)((unsigned long)key->entries & ~JUMP_TYPE_MASK);
}

static inline bool static_key_type(struct static_key *key)
{
 return (unsigned long)key->entries & JUMP_TYPE_MASK;
}

static inline struct static_key *jump_entry_key(struct jump_entry *entry)
{
 return (struct static_key *)((unsigned long)entry->key & ~1UL);
}

static bool jump_entry_branch(struct jump_entry *entry)
{
 return (unsigned long)entry->key & 1UL;
}

static enum jump_label_type jump_label_type(struct jump_entry *entry)
{
 struct static_key *key = jump_entry_key(entry);
 bool enabled = static_key_enabled(key);
 bool branch = jump_entry_branch(entry);


 return enabled ^ branch;
}

static void __jump_label_update(struct static_key *key,
    struct jump_entry *entry,
    struct jump_entry *stop)
{
 for (; (entry < stop) && (jump_entry_key(entry) == key); entry++) {





  if (entry->code && kernel_text_address(entry->code))
   arch_jump_label_transform(entry, jump_label_type(entry));
 }
}

void __init jump_label_init(void)
{
 struct jump_entry *iter_start = __start___jump_table;
 struct jump_entry *iter_stop = __stop___jump_table;
 struct static_key *key = NULL;
 struct jump_entry *iter;

 jump_label_lock();
 jump_label_sort_entries(iter_start, iter_stop);

 for (iter = iter_start; iter < iter_stop; iter++) {
  struct static_key *iterk;


  if (jump_label_type(iter) == JUMP_LABEL_NOP)
   arch_jump_label_transform_static(iter, JUMP_LABEL_NOP);

  iterk = jump_entry_key(iter);
  if (iterk == key)
   continue;

  key = iterk;



  *((unsigned long *)&key->entries) += (unsigned long)iter;
  key->next = NULL;
 }
 static_key_initialized = true;
 jump_label_unlock();
}


static enum jump_label_type jump_label_init_type(struct jump_entry *entry)
{
 struct static_key *key = jump_entry_key(entry);
 bool type = static_key_type(key);
 bool branch = jump_entry_branch(entry);


 return type ^ branch;
}

struct static_key_mod {
 struct static_key_mod *next;
 struct jump_entry *entries;
 struct module *mod;
};

static int __jump_label_mod_text_reserved(void *start, void *end)
{
 struct module *mod;

 mod = __module_text_address((unsigned long)start);
 if (!mod)
  return 0;

 WARN_ON_ONCE(__module_text_address((unsigned long)end) != mod);

 return __jump_label_text_reserved(mod->jump_entries,
    mod->jump_entries + mod->num_jump_entries,
    start, end);
}

static void __jump_label_mod_update(struct static_key *key)
{
 struct static_key_mod *mod;

 for (mod = key->next; mod; mod = mod->next) {
  struct module *m = mod->mod;

  __jump_label_update(key, mod->entries,
        m->jump_entries + m->num_jump_entries);
 }
}
void jump_label_apply_nops(struct module *mod)
{
 struct jump_entry *iter_start = mod->jump_entries;
 struct jump_entry *iter_stop = iter_start + mod->num_jump_entries;
 struct jump_entry *iter;


 if (iter_start == iter_stop)
  return;

 for (iter = iter_start; iter < iter_stop; iter++) {

  if (jump_label_init_type(iter) == JUMP_LABEL_NOP)
   arch_jump_label_transform_static(iter, JUMP_LABEL_NOP);
 }
}

static int jump_label_add_module(struct module *mod)
{
 struct jump_entry *iter_start = mod->jump_entries;
 struct jump_entry *iter_stop = iter_start + mod->num_jump_entries;
 struct jump_entry *iter;
 struct static_key *key = NULL;
 struct static_key_mod *jlm;


 if (iter_start == iter_stop)
  return 0;

 jump_label_sort_entries(iter_start, iter_stop);

 for (iter = iter_start; iter < iter_stop; iter++) {
  struct static_key *iterk;

  iterk = jump_entry_key(iter);
  if (iterk == key)
   continue;

  key = iterk;
  if (within_module(iter->key, mod)) {



   *((unsigned long *)&key->entries) += (unsigned long)iter;
   key->next = NULL;
   continue;
  }
  jlm = kzalloc(sizeof(struct static_key_mod), GFP_KERNEL);
  if (!jlm)
   return -ENOMEM;
  jlm->mod = mod;
  jlm->entries = iter;
  jlm->next = key->next;
  key->next = jlm;


  if (jump_label_type(iter) != jump_label_init_type(iter))
   __jump_label_update(key, iter, iter_stop);
 }

 return 0;
}

static void jump_label_del_module(struct module *mod)
{
 struct jump_entry *iter_start = mod->jump_entries;
 struct jump_entry *iter_stop = iter_start + mod->num_jump_entries;
 struct jump_entry *iter;
 struct static_key *key = NULL;
 struct static_key_mod *jlm, **prev;

 for (iter = iter_start; iter < iter_stop; iter++) {
  if (jump_entry_key(iter) == key)
   continue;

  key = jump_entry_key(iter);

  if (within_module(iter->key, mod))
   continue;

  prev = &key->next;
  jlm = key->next;

  while (jlm && jlm->mod != mod) {
   prev = &jlm->next;
   jlm = jlm->next;
  }

  if (jlm) {
   *prev = jlm->next;
   kfree(jlm);
  }
 }
}

static void jump_label_invalidate_module_init(struct module *mod)
{
 struct jump_entry *iter_start = mod->jump_entries;
 struct jump_entry *iter_stop = iter_start + mod->num_jump_entries;
 struct jump_entry *iter;

 for (iter = iter_start; iter < iter_stop; iter++) {
  if (within_module_init(iter->code, mod))
   iter->code = 0;
 }
}

static int
jump_label_module_notify(struct notifier_block *self, unsigned long val,
    void *data)
{
 struct module *mod = data;
 int ret = 0;

 switch (val) {
 case MODULE_STATE_COMING:
  jump_label_lock();
  ret = jump_label_add_module(mod);
  if (ret)
   jump_label_del_module(mod);
  jump_label_unlock();
  break;
 case MODULE_STATE_GOING:
  jump_label_lock();
  jump_label_del_module(mod);
  jump_label_unlock();
  break;
 case MODULE_STATE_LIVE:
  jump_label_lock();
  jump_label_invalidate_module_init(mod);
  jump_label_unlock();
  break;
 }

 return notifier_from_errno(ret);
}

struct notifier_block jump_label_module_nb = {
 .notifier_call = jump_label_module_notify,
 .priority = 1,
};

static __init int jump_label_init_module(void)
{
 return register_module_notifier(&jump_label_module_nb);
}
early_initcall(jump_label_init_module);

int jump_label_text_reserved(void *start, void *end)
{
 int ret = __jump_label_text_reserved(__start___jump_table,
   __stop___jump_table, start, end);

 if (ret)
  return ret;

 ret = __jump_label_mod_text_reserved(start, end);
 return ret;
}

static void jump_label_update(struct static_key *key)
{
 struct jump_entry *stop = __stop___jump_table;
 struct jump_entry *entry = static_key_entries(key);
 struct module *mod;

 __jump_label_mod_update(key);

 preempt_disable();
 mod = __module_address((unsigned long)key);
 if (mod)
  stop = mod->jump_entries + mod->num_jump_entries;
 preempt_enable();

 if (entry)
  __jump_label_update(key, entry, stop);
}

static DEFINE_STATIC_KEY_TRUE(sk_true);
static DEFINE_STATIC_KEY_FALSE(sk_false);

static __init int jump_label_test(void)
{
 int i;

 for (i = 0; i < 2; i++) {
  WARN_ON(static_key_enabled(&sk_true.key) != true);
  WARN_ON(static_key_enabled(&sk_false.key) != false);

  WARN_ON(!static_branch_likely(&sk_true));
  WARN_ON(!static_branch_unlikely(&sk_true));
  WARN_ON(static_branch_likely(&sk_false));
  WARN_ON(static_branch_unlikely(&sk_false));

  static_branch_disable(&sk_true);
  static_branch_enable(&sk_false);

  WARN_ON(static_key_enabled(&sk_true.key) == true);
  WARN_ON(static_key_enabled(&sk_false.key) == false);

  WARN_ON(static_branch_likely(&sk_true));
  WARN_ON(static_branch_unlikely(&sk_true));
  WARN_ON(!static_branch_likely(&sk_false));
  WARN_ON(!static_branch_unlikely(&sk_false));

  static_branch_enable(&sk_true);
  static_branch_disable(&sk_false);
 }

 return 0;
}
late_initcall(jump_label_test);








extern const unsigned long kallsyms_addresses[] __weak;
extern const int kallsyms_offsets[] __weak;
extern const u8 kallsyms_names[] __weak;





extern const unsigned long kallsyms_num_syms
__attribute__((weak, section(".rodata")));

extern const unsigned long kallsyms_relative_base
__attribute__((weak, section(".rodata")));

extern const u8 kallsyms_token_table[] __weak;
extern const u16 kallsyms_token_index[] __weak;

extern const unsigned long kallsyms_markers[] __weak;

static inline int is_kernel_inittext(unsigned long addr)
{
 if (addr >= (unsigned long)_sinittext
     && addr <= (unsigned long)_einittext)
  return 1;
 return 0;
}

static inline int is_kernel_text(unsigned long addr)
{
 if ((addr >= (unsigned long)_stext && addr <= (unsigned long)_etext) ||
     arch_is_kernel_text(addr))
  return 1;
 return in_gate_area_no_mm(addr);
}

static inline int is_kernel(unsigned long addr)
{
 if (addr >= (unsigned long)_stext && addr <= (unsigned long)_end)
  return 1;
 return in_gate_area_no_mm(addr);
}

static int is_ksym_addr(unsigned long addr)
{
 if (all_var)
  return is_kernel(addr);

 return is_kernel_text(addr) || is_kernel_inittext(addr);
}






static unsigned int kallsyms_expand_symbol(unsigned int off,
        char *result, size_t maxlen)
{
 int len, skipped_first = 0;
 const u8 *tptr, *data;


 data = &kallsyms_names[off];
 len = *data;
 data++;





 off += len + 1;





 while (len) {
  tptr = &kallsyms_token_table[kallsyms_token_index[*data]];
  data++;
  len--;

  while (*tptr) {
   if (skipped_first) {
    if (maxlen <= 1)
     goto tail;
    *result = *tptr;
    result++;
    maxlen--;
   } else
    skipped_first = 1;
   tptr++;
  }
 }

tail:
 if (maxlen)
  *result = '\0';


 return off;
}





static char kallsyms_get_symbol_type(unsigned int off)
{




 return kallsyms_token_table[kallsyms_token_index[kallsyms_names[off + 1]]];
}






static unsigned int get_symbol_offset(unsigned long pos)
{
 const u8 *name;
 int i;





 name = &kallsyms_names[kallsyms_markers[pos >> 8]];







 for (i = 0; i < (pos & 0xFF); i++)
  name = name + (*name) + 1;

 return name - kallsyms_names;
}

static unsigned long kallsyms_sym_address(int idx)
{
 if (!IS_ENABLED(CONFIG_KALLSYMS_BASE_RELATIVE))
  return kallsyms_addresses[idx];


 if (!IS_ENABLED(CONFIG_KALLSYMS_ABSOLUTE_PERCPU))
  return kallsyms_relative_base + (u32)kallsyms_offsets[idx];


 if (kallsyms_offsets[idx] >= 0)
  return kallsyms_offsets[idx];


 return kallsyms_relative_base - 1 - kallsyms_offsets[idx];
}


unsigned long kallsyms_lookup_name(const char *name)
{
 char namebuf[KSYM_NAME_LEN];
 unsigned long i;
 unsigned int off;

 for (i = 0, off = 0; i < kallsyms_num_syms; i++) {
  off = kallsyms_expand_symbol(off, namebuf, ARRAY_SIZE(namebuf));

  if (strcmp(namebuf, name) == 0)
   return kallsyms_sym_address(i);
 }
 return module_kallsyms_lookup_name(name);
}
EXPORT_SYMBOL_GPL(kallsyms_lookup_name);

int kallsyms_on_each_symbol(int (*fn)(void *, const char *, struct module *,
          unsigned long),
       void *data)
{
 char namebuf[KSYM_NAME_LEN];
 unsigned long i;
 unsigned int off;
 int ret;

 for (i = 0, off = 0; i < kallsyms_num_syms; i++) {
  off = kallsyms_expand_symbol(off, namebuf, ARRAY_SIZE(namebuf));
  ret = fn(data, namebuf, NULL, kallsyms_sym_address(i));
  if (ret != 0)
   return ret;
 }
 return module_kallsyms_on_each_symbol(fn, data);
}
EXPORT_SYMBOL_GPL(kallsyms_on_each_symbol);

static unsigned long get_symbol_pos(unsigned long addr,
        unsigned long *symbolsize,
        unsigned long *offset)
{
 unsigned long symbol_start = 0, symbol_end = 0;
 unsigned long i, low, high, mid;


 if (!IS_ENABLED(CONFIG_KALLSYMS_BASE_RELATIVE))
  BUG_ON(!kallsyms_addresses);
 else
  BUG_ON(!kallsyms_offsets);


 low = 0;
 high = kallsyms_num_syms;

 while (high - low > 1) {
  mid = low + (high - low) / 2;
  if (kallsyms_sym_address(mid) <= addr)
   low = mid;
  else
   high = mid;
 }





 while (low && kallsyms_sym_address(low-1) == kallsyms_sym_address(low))
  --low;

 symbol_start = kallsyms_sym_address(low);


 for (i = low + 1; i < kallsyms_num_syms; i++) {
  if (kallsyms_sym_address(i) > symbol_start) {
   symbol_end = kallsyms_sym_address(i);
   break;
  }
 }


 if (!symbol_end) {
  if (is_kernel_inittext(addr))
   symbol_end = (unsigned long)_einittext;
  else if (all_var)
   symbol_end = (unsigned long)_end;
  else
   symbol_end = (unsigned long)_etext;
 }

 if (symbolsize)
  *symbolsize = symbol_end - symbol_start;
 if (offset)
  *offset = addr - symbol_start;

 return low;
}




int kallsyms_lookup_size_offset(unsigned long addr, unsigned long *symbolsize,
    unsigned long *offset)
{
 char namebuf[KSYM_NAME_LEN];
 if (is_ksym_addr(addr))
  return !!get_symbol_pos(addr, symbolsize, offset);

 return !!module_address_lookup(addr, symbolsize, offset, NULL, namebuf);
}
const char *kallsyms_lookup(unsigned long addr,
       unsigned long *symbolsize,
       unsigned long *offset,
       char **modname, char *namebuf)
{
 namebuf[KSYM_NAME_LEN - 1] = 0;
 namebuf[0] = 0;

 if (is_ksym_addr(addr)) {
  unsigned long pos;

  pos = get_symbol_pos(addr, symbolsize, offset);

  kallsyms_expand_symbol(get_symbol_offset(pos),
           namebuf, KSYM_NAME_LEN);
  if (modname)
   *modname = NULL;
  return namebuf;
 }


 return module_address_lookup(addr, symbolsize, offset, modname,
         namebuf);
}

int lookup_symbol_name(unsigned long addr, char *symname)
{
 symname[0] = '\0';
 symname[KSYM_NAME_LEN - 1] = '\0';

 if (is_ksym_addr(addr)) {
  unsigned long pos;

  pos = get_symbol_pos(addr, NULL, NULL);

  kallsyms_expand_symbol(get_symbol_offset(pos),
           symname, KSYM_NAME_LEN);
  return 0;
 }

 return lookup_module_symbol_name(addr, symname);
}

int lookup_symbol_attrs(unsigned long addr, unsigned long *size,
   unsigned long *offset, char *modname, char *name)
{
 name[0] = '\0';
 name[KSYM_NAME_LEN - 1] = '\0';

 if (is_ksym_addr(addr)) {
  unsigned long pos;

  pos = get_symbol_pos(addr, size, offset);

  kallsyms_expand_symbol(get_symbol_offset(pos),
           name, KSYM_NAME_LEN);
  modname[0] = '\0';
  return 0;
 }

 return lookup_module_symbol_attrs(addr, size, offset, modname, name);
}


static int __sprint_symbol(char *buffer, unsigned long address,
      int symbol_offset, int add_offset)
{
 char *modname;
 const char *name;
 unsigned long offset, size;
 int len;

 address += symbol_offset;
 name = kallsyms_lookup(address, &size, &offset, &modname, buffer);
 if (!name)
  return sprintf(buffer, "0x%lx", address - symbol_offset);

 if (name != buffer)
  strcpy(buffer, name);
 len = strlen(buffer);
 offset -= symbol_offset;

 if (add_offset)
  len += sprintf(buffer + len, "+%#lx/%#lx", offset, size);

 if (modname)
  len += sprintf(buffer + len, " [%s]", modname);

 return len;
}
int sprint_symbol(char *buffer, unsigned long address)
{
 return __sprint_symbol(buffer, address, 0, 1);
}
EXPORT_SYMBOL_GPL(sprint_symbol);
int sprint_symbol_no_offset(char *buffer, unsigned long address)
{
 return __sprint_symbol(buffer, address, 0, 0);
}
EXPORT_SYMBOL_GPL(sprint_symbol_no_offset);
int sprint_backtrace(char *buffer, unsigned long address)
{
 return __sprint_symbol(buffer, address, -1, 1);
}


void __print_symbol(const char *fmt, unsigned long address)
{
 char buffer[KSYM_SYMBOL_LEN];

 sprint_symbol(buffer, address);

 printk(fmt, buffer);
}
EXPORT_SYMBOL(__print_symbol);


struct kallsym_iter {
 loff_t pos;
 unsigned long value;
 unsigned int nameoff;
 char type;
 char name[KSYM_NAME_LEN];
 char module_name[MODULE_NAME_LEN];
 int exported;
};

static int get_ksymbol_mod(struct kallsym_iter *iter)
{
 if (module_get_kallsym(iter->pos - kallsyms_num_syms, &iter->value,
    &iter->type, iter->name, iter->module_name,
    &iter->exported) < 0)
  return 0;
 return 1;
}


static unsigned long get_ksymbol_core(struct kallsym_iter *iter)
{
 unsigned off = iter->nameoff;

 iter->module_name[0] = '\0';
 iter->value = kallsyms_sym_address(iter->pos);

 iter->type = kallsyms_get_symbol_type(off);

 off = kallsyms_expand_symbol(off, iter->name, ARRAY_SIZE(iter->name));

 return off - iter->nameoff;
}

static void reset_iter(struct kallsym_iter *iter, loff_t new_pos)
{
 iter->name[0] = '\0';
 iter->nameoff = get_symbol_offset(new_pos);
 iter->pos = new_pos;
}


static int update_iter(struct kallsym_iter *iter, loff_t pos)
{

 if (pos >= kallsyms_num_syms) {
  iter->pos = pos;
  return get_ksymbol_mod(iter);
 }


 if (pos != iter->pos)
  reset_iter(iter, pos);

 iter->nameoff += get_ksymbol_core(iter);
 iter->pos++;

 return 1;
}

static void *s_next(struct seq_file *m, void *p, loff_t *pos)
{
 (*pos)++;

 if (!update_iter(m->private, *pos))
  return NULL;
 return p;
}

static void *s_start(struct seq_file *m, loff_t *pos)
{
 if (!update_iter(m->private, *pos))
  return NULL;
 return m->private;
}

static void s_stop(struct seq_file *m, void *p)
{
}

static int s_show(struct seq_file *m, void *p)
{
 struct kallsym_iter *iter = m->private;


 if (!iter->name[0])
  return 0;

 if (iter->module_name[0]) {
  char type;





  type = iter->exported ? toupper(iter->type) :
     tolower(iter->type);
  seq_printf(m, "%pK %c %s\t[%s]\n", (void *)iter->value,
      type, iter->name, iter->module_name);
 } else
  seq_printf(m, "%pK %c %s\n", (void *)iter->value,
      iter->type, iter->name);
 return 0;
}

static const struct seq_operations kallsyms_op = {
 .start = s_start,
 .next = s_next,
 .stop = s_stop,
 .show = s_show
};

static int kallsyms_open(struct inode *inode, struct file *file)
{





 struct kallsym_iter *iter;
 iter = __seq_open_private(file, &kallsyms_op, sizeof(*iter));
 if (!iter)
  return -ENOMEM;
 reset_iter(iter, 0);

 return 0;
}

const char *kdb_walk_kallsyms(loff_t *pos)
{
 static struct kallsym_iter kdb_walk_kallsyms_iter;
 if (*pos == 0) {
  memset(&kdb_walk_kallsyms_iter, 0,
         sizeof(kdb_walk_kallsyms_iter));
  reset_iter(&kdb_walk_kallsyms_iter, 0);
 }
 while (1) {
  if (!update_iter(&kdb_walk_kallsyms_iter, *pos))
   return NULL;
  ++*pos;

  if (kdb_walk_kallsyms_iter.name[0])
   return kdb_walk_kallsyms_iter.name;
 }
}

static const struct file_operations kallsyms_operations = {
 .open = kallsyms_open,
 .read = seq_read,
 .llseek = seq_lseek,
 .release = seq_release_private,
};

static int __init kallsyms_init(void)
{
 proc_create("kallsyms", 0444, NULL, &kallsyms_operations);
 return 0;
}
device_initcall(kallsyms_init);

static unsigned long cookies[KCMP_TYPES][2] __read_mostly;

static long kptr_obfuscate(long v, int type)
{
 return (v ^ cookies[type][0]) * cookies[type][1];
}







static int kcmp_ptr(void *v1, void *v2, enum kcmp_type type)
{
 long t1, t2;

 t1 = kptr_obfuscate((long)v1, type);
 t2 = kptr_obfuscate((long)v2, type);

 return (t1 < t2) | ((t1 > t2) << 1);
}


static struct file *
get_file_raw_ptr(struct task_struct *task, unsigned int idx)
{
 struct file *file = NULL;

 task_lock(task);
 rcu_read_lock();

 if (task->files)
  file = fcheck_files(task->files, idx);

 rcu_read_unlock();
 task_unlock(task);

 return file;
}

static void kcmp_unlock(struct mutex *m1, struct mutex *m2)
{
 if (likely(m2 != m1))
  mutex_unlock(m2);
 mutex_unlock(m1);
}

static int kcmp_lock(struct mutex *m1, struct mutex *m2)
{
 int err;

 if (m2 > m1)
  swap(m1, m2);

 err = mutex_lock_killable(m1);
 if (!err && likely(m1 != m2)) {
  err = mutex_lock_killable_nested(m2, SINGLE_DEPTH_NESTING);
  if (err)
   mutex_unlock(m1);
 }

 return err;
}

SYSCALL_DEFINE5(kcmp, pid_t, pid1, pid_t, pid2, int, type,
  unsigned long, idx1, unsigned long, idx2)
{
 struct task_struct *task1, *task2;
 int ret;

 rcu_read_lock();




 task1 = find_task_by_vpid(pid1);
 task2 = find_task_by_vpid(pid2);
 if (!task1 || !task2)
  goto err_no_task;

 get_task_struct(task1);
 get_task_struct(task2);

 rcu_read_unlock();




 ret = kcmp_lock(&task1->signal->cred_guard_mutex,
   &task2->signal->cred_guard_mutex);
 if (ret)
  goto err;
 if (!ptrace_may_access(task1, PTRACE_MODE_READ_REALCREDS) ||
     !ptrace_may_access(task2, PTRACE_MODE_READ_REALCREDS)) {
  ret = -EPERM;
  goto err_unlock;
 }

 switch (type) {
 case KCMP_FILE: {
  struct file *filp1, *filp2;

  filp1 = get_file_raw_ptr(task1, idx1);
  filp2 = get_file_raw_ptr(task2, idx2);

  if (filp1 && filp2)
   ret = kcmp_ptr(filp1, filp2, KCMP_FILE);
  else
   ret = -EBADF;
  break;
 }
 case KCMP_VM:
  ret = kcmp_ptr(task1->mm, task2->mm, KCMP_VM);
  break;
 case KCMP_FILES:
  ret = kcmp_ptr(task1->files, task2->files, KCMP_FILES);
  break;
 case KCMP_FS:
  ret = kcmp_ptr(task1->fs, task2->fs, KCMP_FS);
  break;
 case KCMP_SIGHAND:
  ret = kcmp_ptr(task1->sighand, task2->sighand, KCMP_SIGHAND);
  break;
 case KCMP_IO:
  ret = kcmp_ptr(task1->io_context, task2->io_context, KCMP_IO);
  break;
 case KCMP_SYSVSEM:
  ret = kcmp_ptr(task1->sysvsem.undo_list,
          task2->sysvsem.undo_list,
          KCMP_SYSVSEM);
  ret = -EOPNOTSUPP;
  break;
 default:
  ret = -EINVAL;
  break;
 }

err_unlock:
 kcmp_unlock(&task1->signal->cred_guard_mutex,
      &task2->signal->cred_guard_mutex);
err:
 put_task_struct(task1);
 put_task_struct(task2);

 return ret;

err_no_task:
 rcu_read_unlock();
 return -ESRCH;
}

static __init int kcmp_cookies_init(void)
{
 int i;

 get_random_bytes(cookies, sizeof(cookies));

 for (i = 0; i < KCMP_TYPES; i++)
  cookies[i][1] |= (~(~0UL >> 1) | 1);

 return 0;
}
arch_initcall(kcmp_cookies_init);

struct kcov {





 atomic_t refcount;

 spinlock_t lock;
 enum kcov_mode mode;

 unsigned size;

 void *area;

 struct task_struct *t;
};





void notrace __sanitizer_cov_trace_pc(void)
{
 struct task_struct *t;
 enum kcov_mode mode;

 t = current;




 if (!t || in_interrupt())
  return;
 mode = READ_ONCE(t->kcov_mode);
 if (mode == KCOV_MODE_TRACE) {
  unsigned long *area;
  unsigned long pos;
  barrier();
  area = t->kcov_area;

  pos = READ_ONCE(area[0]) + 1;
  if (likely(pos < t->kcov_size)) {
   area[pos] = _RET_IP_;
   WRITE_ONCE(area[0], pos);
  }
 }
}
EXPORT_SYMBOL(__sanitizer_cov_trace_pc);

static void kcov_get(struct kcov *kcov)
{
 atomic_inc(&kcov->refcount);
}

static void kcov_put(struct kcov *kcov)
{
 if (atomic_dec_and_test(&kcov->refcount)) {
  vfree(kcov->area);
  kfree(kcov);
 }
}

void kcov_task_init(struct task_struct *t)
{
 t->kcov_mode = KCOV_MODE_DISABLED;
 t->kcov_size = 0;
 t->kcov_area = NULL;
 t->kcov = NULL;
}

void kcov_task_exit(struct task_struct *t)
{
 struct kcov *kcov;

 kcov = t->kcov;
 if (kcov == NULL)
  return;
 spin_lock(&kcov->lock);
 if (WARN_ON(kcov->t != t)) {
  spin_unlock(&kcov->lock);
  return;
 }

 kcov_task_init(t);
 kcov->t = NULL;
 spin_unlock(&kcov->lock);
 kcov_put(kcov);
}

static int kcov_mmap(struct file *filep, struct vm_area_struct *vma)
{
 int res = 0;
 void *area;
 struct kcov *kcov = vma->vm_file->private_data;
 unsigned long size, off;
 struct page *page;

 area = vmalloc_user(vma->vm_end - vma->vm_start);
 if (!area)
  return -ENOMEM;

 spin_lock(&kcov->lock);
 size = kcov->size * sizeof(unsigned long);
 if (kcov->mode == KCOV_MODE_DISABLED || vma->vm_pgoff != 0 ||
     vma->vm_end - vma->vm_start != size) {
  res = -EINVAL;
  goto exit;
 }
 if (!kcov->area) {
  kcov->area = area;
  vma->vm_flags |= VM_DONTEXPAND;
  spin_unlock(&kcov->lock);
  for (off = 0; off < size; off += PAGE_SIZE) {
   page = vmalloc_to_page(kcov->area + off);
   if (vm_insert_page(vma, vma->vm_start + off, page))
    WARN_ONCE(1, "vm_insert_page() failed");
  }
  return 0;
 }
exit:
 spin_unlock(&kcov->lock);
 vfree(area);
 return res;
}

static int kcov_open(struct inode *inode, struct file *filep)
{
 struct kcov *kcov;

 kcov = kzalloc(sizeof(*kcov), GFP_KERNEL);
 if (!kcov)
  return -ENOMEM;
 atomic_set(&kcov->refcount, 1);
 spin_lock_init(&kcov->lock);
 filep->private_data = kcov;
 return nonseekable_open(inode, filep);
}

static int kcov_close(struct inode *inode, struct file *filep)
{
 kcov_put(filep->private_data);
 return 0;
}

static int kcov_ioctl_locked(struct kcov *kcov, unsigned int cmd,
        unsigned long arg)
{
 struct task_struct *t;
 unsigned long size, unused;

 switch (cmd) {
 case KCOV_INIT_TRACE:




  if (kcov->mode != KCOV_MODE_DISABLED)
   return -EBUSY;





  size = arg;
  if (size < 2 || size > INT_MAX / sizeof(unsigned long))
   return -EINVAL;
  kcov->size = size;
  kcov->mode = KCOV_MODE_TRACE;
  return 0;
 case KCOV_ENABLE:







  unused = arg;
  if (unused != 0 || kcov->mode == KCOV_MODE_DISABLED ||
      kcov->area == NULL)
   return -EINVAL;
  if (kcov->t != NULL)
   return -EBUSY;
  t = current;

  t->kcov_size = kcov->size;
  t->kcov_area = kcov->area;

  barrier();
  WRITE_ONCE(t->kcov_mode, kcov->mode);
  t->kcov = kcov;
  kcov->t = t;

  kcov_get(kcov);
  return 0;
 case KCOV_DISABLE:

  unused = arg;
  if (unused != 0 || current->kcov != kcov)
   return -EINVAL;
  t = current;
  if (WARN_ON(kcov->t != t))
   return -EINVAL;
  kcov_task_init(t);
  kcov->t = NULL;
  kcov_put(kcov);
  return 0;
 default:
  return -ENOTTY;
 }
}

static long kcov_ioctl(struct file *filep, unsigned int cmd, unsigned long arg)
{
 struct kcov *kcov;
 int res;

 kcov = filep->private_data;
 spin_lock(&kcov->lock);
 res = kcov_ioctl_locked(kcov, cmd, arg);
 spin_unlock(&kcov->lock);
 return res;
}

static const struct file_operations kcov_fops = {
 .open = kcov_open,
 .unlocked_ioctl = kcov_ioctl,
 .mmap = kcov_mmap,
 .release = kcov_close,
};

static int __init kcov_init(void)
{





 if (!debugfs_create_file_unsafe("kcov", 0600, NULL, NULL, &kcov_fops)) {
  pr_err("failed to create kcov in debugfs\n");
  return -ENOMEM;
 }
 return 0;
}

device_initcall(kcov_init);



static int copy_user_segment_list(struct kimage *image,
      unsigned long nr_segments,
      struct kexec_segment __user *segments)
{
 int ret;
 size_t segment_bytes;


 image->nr_segments = nr_segments;
 segment_bytes = nr_segments * sizeof(*segments);
 ret = copy_from_user(image->segment, segments, segment_bytes);
 if (ret)
  ret = -EFAULT;

 return ret;
}

static int kimage_alloc_init(struct kimage **rimage, unsigned long entry,
        unsigned long nr_segments,
        struct kexec_segment __user *segments,
        unsigned long flags)
{
 int ret;
 struct kimage *image;
 bool kexec_on_panic = flags & KEXEC_ON_CRASH;

 if (kexec_on_panic) {

  if ((entry < crashk_res.start) || (entry > crashk_res.end))
   return -EADDRNOTAVAIL;
 }


 image = do_kimage_alloc_init();
 if (!image)
  return -ENOMEM;

 image->start = entry;

 ret = copy_user_segment_list(image, nr_segments, segments);
 if (ret)
  goto out_free_image;

 if (kexec_on_panic) {

  image->control_page = crashk_res.start;
  image->type = KEXEC_TYPE_CRASH;
 }

 ret = sanity_check_segment_list(image);
 if (ret)
  goto out_free_image;






 ret = -ENOMEM;
 image->control_code_page = kimage_alloc_control_pages(image,
        get_order(KEXEC_CONTROL_PAGE_SIZE));
 if (!image->control_code_page) {
  pr_err("Could not allocate control_code_buffer\n");
  goto out_free_image;
 }

 if (!kexec_on_panic) {
  image->swap_page = kimage_alloc_control_pages(image, 0);
  if (!image->swap_page) {
   pr_err("Could not allocate swap buffer\n");
   goto out_free_control_pages;
  }
 }

 *rimage = image;
 return 0;
out_free_control_pages:
 kimage_free_page_list(&image->control_pages);
out_free_image:
 kfree(image);
 return ret;
}

static int do_kexec_load(unsigned long entry, unsigned long nr_segments,
  struct kexec_segment __user *segments, unsigned long flags)
{
 struct kimage **dest_image, *image;
 unsigned long i;
 int ret;

 if (flags & KEXEC_ON_CRASH) {
  dest_image = &kexec_crash_image;
  if (kexec_crash_image)
   arch_kexec_unprotect_crashkres();
 } else {
  dest_image = &kexec_image;
 }

 if (nr_segments == 0) {

  kimage_free(xchg(dest_image, NULL));
  return 0;
 }
 if (flags & KEXEC_ON_CRASH) {





  kimage_free(xchg(&kexec_crash_image, NULL));
 }

 ret = kimage_alloc_init(&image, entry, nr_segments, segments, flags);
 if (ret)
  return ret;

 if (flags & KEXEC_PRESERVE_CONTEXT)
  image->preserve_context = 1;

 ret = machine_kexec_prepare(image);
 if (ret)
  goto out;

 for (i = 0; i < nr_segments; i++) {
  ret = kimage_load_segment(image, &image->segment[i]);
  if (ret)
   goto out;
 }

 kimage_terminate(image);


 image = xchg(dest_image, image);

out:
 if ((flags & KEXEC_ON_CRASH) && kexec_crash_image)
  arch_kexec_protect_crashkres();

 kimage_free(image);
 return ret;
}
SYSCALL_DEFINE4(kexec_load, unsigned long, entry, unsigned long, nr_segments,
  struct kexec_segment __user *, segments, unsigned long, flags)
{
 int result;


 if (!capable(CAP_SYS_BOOT) || kexec_load_disabled)
  return -EPERM;





 if ((flags & KEXEC_FLAGS) != (flags & ~KEXEC_ARCH_MASK))
  return -EINVAL;


 if (((flags & KEXEC_ARCH_MASK) != KEXEC_ARCH) &&
  ((flags & KEXEC_ARCH_MASK) != KEXEC_ARCH_DEFAULT))
  return -EINVAL;




 if (nr_segments > KEXEC_SEGMENT_MAX)
  return -EINVAL;
 if (!mutex_trylock(&kexec_mutex))
  return -EBUSY;

 result = do_kexec_load(entry, nr_segments, segments, flags);

 mutex_unlock(&kexec_mutex);

 return result;
}

COMPAT_SYSCALL_DEFINE4(kexec_load, compat_ulong_t, entry,
         compat_ulong_t, nr_segments,
         struct compat_kexec_segment __user *, segments,
         compat_ulong_t, flags)
{
 struct compat_kexec_segment in;
 struct kexec_segment out, __user *ksegments;
 unsigned long i, result;




 if ((flags & KEXEC_ARCH_MASK) == KEXEC_ARCH_DEFAULT)
  return -EINVAL;

 if (nr_segments > KEXEC_SEGMENT_MAX)
  return -EINVAL;

 ksegments = compat_alloc_user_space(nr_segments * sizeof(out));
 for (i = 0; i < nr_segments; i++) {
  result = copy_from_user(&in, &segments[i], sizeof(in));
  if (result)
   return -EFAULT;

  out.buf = compat_ptr(in.buf);
  out.bufsz = in.bufsz;
  out.mem = in.mem;
  out.memsz = in.memsz;

  result = copy_to_user(&ksegments[i], &out, sizeof(out));
  if (result)
   return -EFAULT;
 }

 return sys_kexec_load(entry, nr_segments, ksegments, flags);
}




DEFINE_MUTEX(kexec_mutex);


note_buf_t __percpu *crash_notes;


static unsigned char vmcoreinfo_data[VMCOREINFO_BYTES];
u32 vmcoreinfo_note[VMCOREINFO_NOTE_SIZE/4];
size_t vmcoreinfo_size;
size_t vmcoreinfo_max_size = sizeof(vmcoreinfo_data);


bool kexec_in_progress = false;



struct resource crashk_res = {
 .name = "Crash kernel",
 .start = 0,
 .end = 0,
 .flags = IORESOURCE_BUSY | IORESOURCE_SYSTEM_RAM,
 .desc = IORES_DESC_CRASH_KERNEL
};
struct resource crashk_low_res = {
 .name = "Crash kernel",
 .start = 0,
 .end = 0,
 .flags = IORESOURCE_BUSY | IORESOURCE_SYSTEM_RAM,
 .desc = IORES_DESC_CRASH_KERNEL
};

int kexec_should_crash(struct task_struct *p)
{





 if (crash_kexec_post_notifiers)
  return 0;




 if (in_interrupt() || !p->pid || is_global_init(p) || panic_on_oops)
  return 1;
 return 0;
}

static struct page *kimage_alloc_page(struct kimage *image,
           gfp_t gfp_mask,
           unsigned long dest);

int sanity_check_segment_list(struct kimage *image)
{
 int result, i;
 unsigned long nr_segments = image->nr_segments;
 result = -EADDRNOTAVAIL;
 for (i = 0; i < nr_segments; i++) {
  unsigned long mstart, mend;

  mstart = image->segment[i].mem;
  mend = mstart + image->segment[i].memsz;
  if ((mstart & ~PAGE_MASK) || (mend & ~PAGE_MASK))
   return result;
  if (mend >= KEXEC_DESTINATION_MEMORY_LIMIT)
   return result;
 }






 result = -EINVAL;
 for (i = 0; i < nr_segments; i++) {
  unsigned long mstart, mend;
  unsigned long j;

  mstart = image->segment[i].mem;
  mend = mstart + image->segment[i].memsz;
  for (j = 0; j < i; j++) {
   unsigned long pstart, pend;

   pstart = image->segment[j].mem;
   pend = pstart + image->segment[j].memsz;

   if ((mend > pstart) && (mstart < pend))
    return result;
  }
 }






 result = -EINVAL;
 for (i = 0; i < nr_segments; i++) {
  if (image->segment[i].bufsz > image->segment[i].memsz)
   return result;
 }
 if (image->type == KEXEC_TYPE_CRASH) {
  result = -EADDRNOTAVAIL;
  for (i = 0; i < nr_segments; i++) {
   unsigned long mstart, mend;

   mstart = image->segment[i].mem;
   mend = mstart + image->segment[i].memsz - 1;

   if ((mstart < crashk_res.start) ||
       (mend > crashk_res.end))
    return result;
  }
 }

 return 0;
}

struct kimage *do_kimage_alloc_init(void)
{
 struct kimage *image;


 image = kzalloc(sizeof(*image), GFP_KERNEL);
 if (!image)
  return NULL;

 image->head = 0;
 image->entry = &image->head;
 image->last_entry = &image->head;
 image->control_page = ~0;
 image->type = KEXEC_TYPE_DEFAULT;


 INIT_LIST_HEAD(&image->control_pages);


 INIT_LIST_HEAD(&image->dest_pages);


 INIT_LIST_HEAD(&image->unusable_pages);

 return image;
}

int kimage_is_destination_range(struct kimage *image,
     unsigned long start,
     unsigned long end)
{
 unsigned long i;

 for (i = 0; i < image->nr_segments; i++) {
  unsigned long mstart, mend;

  mstart = image->segment[i].mem;
  mend = mstart + image->segment[i].memsz;
  if ((end > mstart) && (start < mend))
   return 1;
 }

 return 0;
}

static struct page *kimage_alloc_pages(gfp_t gfp_mask, unsigned int order)
{
 struct page *pages;

 pages = alloc_pages(gfp_mask, order);
 if (pages) {
  unsigned int count, i;

  pages->mapping = NULL;
  set_page_private(pages, order);
  count = 1 << order;
  for (i = 0; i < count; i++)
   SetPageReserved(pages + i);
 }

 return pages;
}

static void kimage_free_pages(struct page *page)
{
 unsigned int order, count, i;

 order = page_private(page);
 count = 1 << order;
 for (i = 0; i < count; i++)
  ClearPageReserved(page + i);
 __free_pages(page, order);
}

void kimage_free_page_list(struct list_head *list)
{
 struct page *page, *next;

 list_for_each_entry_safe(page, next, list, lru) {
  list_del(&page->lru);
  kimage_free_pages(page);
 }
}

static struct page *kimage_alloc_normal_control_pages(struct kimage *image,
       unsigned int order)
{
 struct list_head extra_pages;
 struct page *pages;
 unsigned int count;

 count = 1 << order;
 INIT_LIST_HEAD(&extra_pages);




 do {
  unsigned long pfn, epfn, addr, eaddr;

  pages = kimage_alloc_pages(KEXEC_CONTROL_MEMORY_GFP, order);
  if (!pages)
   break;
  pfn = page_to_pfn(pages);
  epfn = pfn + count;
  addr = pfn << PAGE_SHIFT;
  eaddr = epfn << PAGE_SHIFT;
  if ((epfn >= (KEXEC_CONTROL_MEMORY_LIMIT >> PAGE_SHIFT)) ||
         kimage_is_destination_range(image, addr, eaddr)) {
   list_add(&pages->lru, &extra_pages);
   pages = NULL;
  }
 } while (!pages);

 if (pages) {

  list_add(&pages->lru, &image->control_pages);







 }







 kimage_free_page_list(&extra_pages);

 return pages;
}

static struct page *kimage_alloc_crash_control_pages(struct kimage *image,
            unsigned int order)
{
 unsigned long hole_start, hole_end, size;
 struct page *pages;

 pages = NULL;
 size = (1 << order) << PAGE_SHIFT;
 hole_start = (image->control_page + (size - 1)) & ~(size - 1);
 hole_end = hole_start + size - 1;
 while (hole_end <= crashk_res.end) {
  unsigned long i;

  if (hole_end > KEXEC_CRASH_CONTROL_MEMORY_LIMIT)
   break;

  for (i = 0; i < image->nr_segments; i++) {
   unsigned long mstart, mend;

   mstart = image->segment[i].mem;
   mend = mstart + image->segment[i].memsz - 1;
   if ((hole_end >= mstart) && (hole_start <= mend)) {

    hole_start = (mend + (size - 1)) & ~(size - 1);
    hole_end = hole_start + size - 1;
    break;
   }
  }

  if (i == image->nr_segments) {
   pages = pfn_to_page(hole_start >> PAGE_SHIFT);
   image->control_page = hole_end;
   break;
  }
 }

 return pages;
}


struct page *kimage_alloc_control_pages(struct kimage *image,
      unsigned int order)
{
 struct page *pages = NULL;

 switch (image->type) {
 case KEXEC_TYPE_DEFAULT:
  pages = kimage_alloc_normal_control_pages(image, order);
  break;
 case KEXEC_TYPE_CRASH:
  pages = kimage_alloc_crash_control_pages(image, order);
  break;
 }

 return pages;
}

static int kimage_add_entry(struct kimage *image, kimage_entry_t entry)
{
 if (*image->entry != 0)
  image->entry++;

 if (image->entry == image->last_entry) {
  kimage_entry_t *ind_page;
  struct page *page;

  page = kimage_alloc_page(image, GFP_KERNEL, KIMAGE_NO_DEST);
  if (!page)
   return -ENOMEM;

  ind_page = page_address(page);
  *image->entry = virt_to_phys(ind_page) | IND_INDIRECTION;
  image->entry = ind_page;
  image->last_entry = ind_page +
          ((PAGE_SIZE/sizeof(kimage_entry_t)) - 1);
 }
 *image->entry = entry;
 image->entry++;
 *image->entry = 0;

 return 0;
}

static int kimage_set_destination(struct kimage *image,
       unsigned long destination)
{
 int result;

 destination &= PAGE_MASK;
 result = kimage_add_entry(image, destination | IND_DESTINATION);

 return result;
}


static int kimage_add_page(struct kimage *image, unsigned long page)
{
 int result;

 page &= PAGE_MASK;
 result = kimage_add_entry(image, page | IND_SOURCE);

 return result;
}


static void kimage_free_extra_pages(struct kimage *image)
{

 kimage_free_page_list(&image->dest_pages);


 kimage_free_page_list(&image->unusable_pages);

}
void kimage_terminate(struct kimage *image)
{
 if (*image->entry != 0)
  image->entry++;

 *image->entry = IND_DONE;
}

 for (ptr = &image->head; (entry = *ptr) && !(entry & IND_DONE); \
  ptr = (entry & IND_INDIRECTION) ? \
   phys_to_virt((entry & PAGE_MASK)) : ptr + 1)

static void kimage_free_entry(kimage_entry_t entry)
{
 struct page *page;

 page = pfn_to_page(entry >> PAGE_SHIFT);
 kimage_free_pages(page);
}

void kimage_free(struct kimage *image)
{
 kimage_entry_t *ptr, entry;
 kimage_entry_t ind = 0;

 if (!image)
  return;

 kimage_free_extra_pages(image);
 for_each_kimage_entry(image, ptr, entry) {
  if (entry & IND_INDIRECTION) {

   if (ind & IND_INDIRECTION)
    kimage_free_entry(ind);



   ind = entry;
  } else if (entry & IND_SOURCE)
   kimage_free_entry(entry);
 }

 if (ind & IND_INDIRECTION)
  kimage_free_entry(ind);


 machine_kexec_cleanup(image);


 kimage_free_page_list(&image->control_pages);





 if (image->file_mode)
  kimage_file_post_load_cleanup(image);

 kfree(image);
}

static kimage_entry_t *kimage_dst_used(struct kimage *image,
     unsigned long page)
{
 kimage_entry_t *ptr, entry;
 unsigned long destination = 0;

 for_each_kimage_entry(image, ptr, entry) {
  if (entry & IND_DESTINATION)
   destination = entry & PAGE_MASK;
  else if (entry & IND_SOURCE) {
   if (page == destination)
    return ptr;
   destination += PAGE_SIZE;
  }
 }

 return NULL;
}

static struct page *kimage_alloc_page(struct kimage *image,
     gfp_t gfp_mask,
     unsigned long destination)
{
 struct page *page;
 unsigned long addr;





 list_for_each_entry(page, &image->dest_pages, lru) {
  addr = page_to_pfn(page) << PAGE_SHIFT;
  if (addr == destination) {
   list_del(&page->lru);
   return page;
  }
 }
 page = NULL;
 while (1) {
  kimage_entry_t *old;


  page = kimage_alloc_pages(gfp_mask, 0);
  if (!page)
   return NULL;

  if (page_to_pfn(page) >
    (KEXEC_SOURCE_MEMORY_LIMIT >> PAGE_SHIFT)) {
   list_add(&page->lru, &image->unusable_pages);
   continue;
  }
  addr = page_to_pfn(page) << PAGE_SHIFT;


  if (addr == destination)
   break;


  if (!kimage_is_destination_range(image, addr,
        addr + PAGE_SIZE))
   break;






  old = kimage_dst_used(image, addr);
  if (old) {

   unsigned long old_addr;
   struct page *old_page;

   old_addr = *old & PAGE_MASK;
   old_page = pfn_to_page(old_addr >> PAGE_SHIFT);
   copy_highpage(page, old_page);
   *old = addr | (*old & ~PAGE_MASK);





   if (!(gfp_mask & __GFP_HIGHMEM) &&
       PageHighMem(old_page)) {
    kimage_free_pages(old_page);
    continue;
   }
   addr = old_addr;
   page = old_page;
   break;
  }

  list_add(&page->lru, &image->dest_pages);
 }

 return page;
}

static int kimage_load_normal_segment(struct kimage *image,
      struct kexec_segment *segment)
{
 unsigned long maddr;
 size_t ubytes, mbytes;
 int result;
 unsigned char __user *buf = NULL;
 unsigned char *kbuf = NULL;

 result = 0;
 if (image->file_mode)
  kbuf = segment->kbuf;
 else
  buf = segment->buf;
 ubytes = segment->bufsz;
 mbytes = segment->memsz;
 maddr = segment->mem;

 result = kimage_set_destination(image, maddr);
 if (result < 0)
  goto out;

 while (mbytes) {
  struct page *page;
  char *ptr;
  size_t uchunk, mchunk;

  page = kimage_alloc_page(image, GFP_HIGHUSER, maddr);
  if (!page) {
   result = -ENOMEM;
   goto out;
  }
  result = kimage_add_page(image, page_to_pfn(page)
        << PAGE_SHIFT);
  if (result < 0)
   goto out;

  ptr = kmap(page);

  clear_page(ptr);
  ptr += maddr & ~PAGE_MASK;
  mchunk = min_t(size_t, mbytes,
    PAGE_SIZE - (maddr & ~PAGE_MASK));
  uchunk = min(ubytes, mchunk);


  if (image->file_mode)
   memcpy(ptr, kbuf, uchunk);
  else
   result = copy_from_user(ptr, buf, uchunk);
  kunmap(page);
  if (result) {
   result = -EFAULT;
   goto out;
  }
  ubytes -= uchunk;
  maddr += mchunk;
  if (image->file_mode)
   kbuf += mchunk;
  else
   buf += mchunk;
  mbytes -= mchunk;
 }
out:
 return result;
}

static int kimage_load_crash_segment(struct kimage *image,
     struct kexec_segment *segment)
{




 unsigned long maddr;
 size_t ubytes, mbytes;
 int result;
 unsigned char __user *buf = NULL;
 unsigned char *kbuf = NULL;

 result = 0;
 if (image->file_mode)
  kbuf = segment->kbuf;
 else
  buf = segment->buf;
 ubytes = segment->bufsz;
 mbytes = segment->memsz;
 maddr = segment->mem;
 while (mbytes) {
  struct page *page;
  char *ptr;
  size_t uchunk, mchunk;

  page = pfn_to_page(maddr >> PAGE_SHIFT);
  if (!page) {
   result = -ENOMEM;
   goto out;
  }
  ptr = kmap(page);
  ptr += maddr & ~PAGE_MASK;
  mchunk = min_t(size_t, mbytes,
    PAGE_SIZE - (maddr & ~PAGE_MASK));
  uchunk = min(ubytes, mchunk);
  if (mchunk > uchunk) {

   memset(ptr + uchunk, 0, mchunk - uchunk);
  }


  if (image->file_mode)
   memcpy(ptr, kbuf, uchunk);
  else
   result = copy_from_user(ptr, buf, uchunk);
  kexec_flush_icache_page(page);
  kunmap(page);
  if (result) {
   result = -EFAULT;
   goto out;
  }
  ubytes -= uchunk;
  maddr += mchunk;
  if (image->file_mode)
   kbuf += mchunk;
  else
   buf += mchunk;
  mbytes -= mchunk;
 }
out:
 return result;
}

int kimage_load_segment(struct kimage *image,
    struct kexec_segment *segment)
{
 int result = -ENOMEM;

 switch (image->type) {
 case KEXEC_TYPE_DEFAULT:
  result = kimage_load_normal_segment(image, segment);
  break;
 case KEXEC_TYPE_CRASH:
  result = kimage_load_crash_segment(image, segment);
  break;
 }

 return result;
}

struct kimage *kexec_image;
struct kimage *kexec_crash_image;
int kexec_load_disabled;






void __crash_kexec(struct pt_regs *regs)
{
 if (mutex_trylock(&kexec_mutex)) {
  if (kexec_crash_image) {
   struct pt_regs fixed_regs;

   crash_setup_regs(&fixed_regs, regs);
   crash_save_vmcoreinfo();
   machine_crash_shutdown(&fixed_regs);
   machine_kexec(kexec_crash_image);
  }
  mutex_unlock(&kexec_mutex);
 }
}

void crash_kexec(struct pt_regs *regs)
{
 int old_cpu, this_cpu;






 this_cpu = raw_smp_processor_id();
 old_cpu = atomic_cmpxchg(&panic_cpu, PANIC_CPU_INVALID, this_cpu);
 if (old_cpu == PANIC_CPU_INVALID) {

  printk_nmi_flush_on_panic();
  __crash_kexec(regs);





  atomic_set(&panic_cpu, PANIC_CPU_INVALID);
 }
}

size_t crash_get_memory_size(void)
{
 size_t size = 0;

 mutex_lock(&kexec_mutex);
 if (crashk_res.end != crashk_res.start)
  size = resource_size(&crashk_res);
 mutex_unlock(&kexec_mutex);
 return size;
}

void __weak crash_free_reserved_phys_range(unsigned long begin,
        unsigned long end)
{
 unsigned long addr;

 for (addr = begin; addr < end; addr += PAGE_SIZE)
  free_reserved_page(pfn_to_page(addr >> PAGE_SHIFT));
}

int crash_shrink_memory(unsigned long new_size)
{
 int ret = 0;
 unsigned long start, end;
 unsigned long old_size;
 struct resource *ram_res;

 mutex_lock(&kexec_mutex);

 if (kexec_crash_image) {
  ret = -ENOENT;
  goto unlock;
 }
 start = crashk_res.start;
 end = crashk_res.end;
 old_size = (end == 0) ? 0 : end - start + 1;
 if (new_size >= old_size) {
  ret = (new_size == old_size) ? 0 : -EINVAL;
  goto unlock;
 }

 ram_res = kzalloc(sizeof(*ram_res), GFP_KERNEL);
 if (!ram_res) {
  ret = -ENOMEM;
  goto unlock;
 }

 start = roundup(start, KEXEC_CRASH_MEM_ALIGN);
 end = roundup(start + new_size, KEXEC_CRASH_MEM_ALIGN);

 crash_free_reserved_phys_range(end, crashk_res.end);

 if ((start == end) && (crashk_res.parent != NULL))
  release_resource(&crashk_res);

 ram_res->start = end;
 ram_res->end = crashk_res.end;
 ram_res->flags = IORESOURCE_BUSY | IORESOURCE_SYSTEM_RAM;
 ram_res->name = "System RAM";

 crashk_res.end = end - 1;

 insert_resource(&iomem_resource, ram_res);

unlock:
 mutex_unlock(&kexec_mutex);
 return ret;
}

static u32 *append_elf_note(u32 *buf, char *name, unsigned type, void *data,
       size_t data_len)
{
 struct elf_note note;

 note.n_namesz = strlen(name) + 1;
 note.n_descsz = data_len;
 note.n_type = type;
 memcpy(buf, &note, sizeof(note));
 buf += (sizeof(note) + 3)/4;
 memcpy(buf, name, note.n_namesz);
 buf += (note.n_namesz + 3)/4;
 memcpy(buf, data, note.n_descsz);
 buf += (note.n_descsz + 3)/4;

 return buf;
}

static void final_note(u32 *buf)
{
 struct elf_note note;

 note.n_namesz = 0;
 note.n_descsz = 0;
 note.n_type = 0;
 memcpy(buf, &note, sizeof(note));
}

void crash_save_cpu(struct pt_regs *regs, int cpu)
{
 struct elf_prstatus prstatus;
 u32 *buf;

 if ((cpu < 0) || (cpu >= nr_cpu_ids))
  return;
 buf = (u32 *)per_cpu_ptr(crash_notes, cpu);
 if (!buf)
  return;
 memset(&prstatus, 0, sizeof(prstatus));
 prstatus.pr_pid = current->pid;
 elf_core_copy_kernel_regs(&prstatus.pr_reg, regs);
 buf = append_elf_note(buf, KEXEC_CORE_NOTE_NAME, NT_PRSTATUS,
         &prstatus, sizeof(prstatus));
 final_note(buf);
}

static int __init crash_notes_memory_init(void)
{

 size_t size, align;
 size = sizeof(note_buf_t);
 align = min(roundup_pow_of_two(sizeof(note_buf_t)), PAGE_SIZE);





 BUILD_BUG_ON(size > PAGE_SIZE);

 crash_notes = __alloc_percpu(size, align);
 if (!crash_notes) {
  pr_warn("Memory allocation for saving cpu register states failed\n");
  return -ENOMEM;
 }
 return 0;
}
subsys_initcall(crash_notes_memory_init);
static int __init parse_crashkernel_mem(char *cmdline,
     unsigned long long system_ram,
     unsigned long long *crash_size,
     unsigned long long *crash_base)
{
 char *cur = cmdline, *tmp;


 do {
  unsigned long long start, end = ULLONG_MAX, size;


  start = memparse(cur, &tmp);
  if (cur == tmp) {
   pr_warn("crashkernel: Memory value expected\n");
   return -EINVAL;
  }
  cur = tmp;
  if (*cur != '-') {
   pr_warn("crashkernel: '-' expected\n");
   return -EINVAL;
  }
  cur++;


  if (*cur != ':') {
   end = memparse(cur, &tmp);
   if (cur == tmp) {
    pr_warn("crashkernel: Memory value expected\n");
    return -EINVAL;
   }
   cur = tmp;
   if (end <= start) {
    pr_warn("crashkernel: end <= start\n");
    return -EINVAL;
   }
  }

  if (*cur != ':') {
   pr_warn("crashkernel: ':' expected\n");
   return -EINVAL;
  }
  cur++;

  size = memparse(cur, &tmp);
  if (cur == tmp) {
   pr_warn("Memory value expected\n");
   return -EINVAL;
  }
  cur = tmp;
  if (size >= system_ram) {
   pr_warn("crashkernel: invalid size\n");
   return -EINVAL;
  }


  if (system_ram >= start && system_ram < end) {
   *crash_size = size;
   break;
  }
 } while (*cur++ == ',');

 if (*crash_size > 0) {
  while (*cur && *cur != ' ' && *cur != '@')
   cur++;
  if (*cur == '@') {
   cur++;
   *crash_base = memparse(cur, &tmp);
   if (cur == tmp) {
    pr_warn("Memory value expected after '@'\n");
    return -EINVAL;
   }
  }
 }

 return 0;
}
static int __init parse_crashkernel_simple(char *cmdline,
        unsigned long long *crash_size,
        unsigned long long *crash_base)
{
 char *cur = cmdline;

 *crash_size = memparse(cmdline, &cur);
 if (cmdline == cur) {
  pr_warn("crashkernel: memory value expected\n");
  return -EINVAL;
 }

 if (*cur == '@')
  *crash_base = memparse(cur+1, &cur);
 else if (*cur != ' ' && *cur != '\0') {
  pr_warn("crashkernel: unrecognized char: %c\n", *cur);
  return -EINVAL;
 }

 return 0;
}

static __initdata char *suffix_tbl[] = {
 [SUFFIX_HIGH] = ",high",
 [SUFFIX_LOW] = ",low",
 [SUFFIX_NULL] = NULL,
};
static int __init parse_crashkernel_suffix(char *cmdline,
        unsigned long long *crash_size,
        const char *suffix)
{
 char *cur = cmdline;

 *crash_size = memparse(cmdline, &cur);
 if (cmdline == cur) {
  pr_warn("crashkernel: memory value expected\n");
  return -EINVAL;
 }


 if (strncmp(cur, suffix, strlen(suffix))) {
  pr_warn("crashkernel: unrecognized char: %c\n", *cur);
  return -EINVAL;
 }
 cur += strlen(suffix);
 if (*cur != ' ' && *cur != '\0') {
  pr_warn("crashkernel: unrecognized char: %c\n", *cur);
  return -EINVAL;
 }

 return 0;
}

static __init char *get_last_crashkernel(char *cmdline,
        const char *name,
        const char *suffix)
{
 char *p = cmdline, *ck_cmdline = NULL;


 p = strstr(p, name);
 while (p) {
  char *end_p = strchr(p, ' ');
  char *q;

  if (!end_p)
   end_p = p + strlen(p);

  if (!suffix) {
   int i;


   for (i = 0; suffix_tbl[i]; i++) {
    q = end_p - strlen(suffix_tbl[i]);
    if (!strncmp(q, suffix_tbl[i],
          strlen(suffix_tbl[i])))
     goto next;
   }
   ck_cmdline = p;
  } else {
   q = end_p - strlen(suffix);
   if (!strncmp(q, suffix, strlen(suffix)))
    ck_cmdline = p;
  }
next:
  p = strstr(p+1, name);
 }

 if (!ck_cmdline)
  return NULL;

 return ck_cmdline;
}

static int __init __parse_crashkernel(char *cmdline,
        unsigned long long system_ram,
        unsigned long long *crash_size,
        unsigned long long *crash_base,
        const char *name,
        const char *suffix)
{
 char *first_colon, *first_space;
 char *ck_cmdline;

 BUG_ON(!crash_size || !crash_base);
 *crash_size = 0;
 *crash_base = 0;

 ck_cmdline = get_last_crashkernel(cmdline, name, suffix);

 if (!ck_cmdline)
  return -EINVAL;

 ck_cmdline += strlen(name);

 if (suffix)
  return parse_crashkernel_suffix(ck_cmdline, crash_size,
    suffix);




 first_colon = strchr(ck_cmdline, ':');
 first_space = strchr(ck_cmdline, ' ');
 if (first_colon && (!first_space || first_colon < first_space))
  return parse_crashkernel_mem(ck_cmdline, system_ram,
    crash_size, crash_base);

 return parse_crashkernel_simple(ck_cmdline, crash_size, crash_base);
}





int __init parse_crashkernel(char *cmdline,
        unsigned long long system_ram,
        unsigned long long *crash_size,
        unsigned long long *crash_base)
{
 return __parse_crashkernel(cmdline, system_ram, crash_size, crash_base,
     "crashkernel=", NULL);
}

int __init parse_crashkernel_high(char *cmdline,
        unsigned long long system_ram,
        unsigned long long *crash_size,
        unsigned long long *crash_base)
{
 return __parse_crashkernel(cmdline, system_ram, crash_size, crash_base,
    "crashkernel=", suffix_tbl[SUFFIX_HIGH]);
}

int __init parse_crashkernel_low(char *cmdline,
        unsigned long long system_ram,
        unsigned long long *crash_size,
        unsigned long long *crash_base)
{
 return __parse_crashkernel(cmdline, system_ram, crash_size, crash_base,
    "crashkernel=", suffix_tbl[SUFFIX_LOW]);
}

static void update_vmcoreinfo_note(void)
{
 u32 *buf = vmcoreinfo_note;

 if (!vmcoreinfo_size)
  return;
 buf = append_elf_note(buf, VMCOREINFO_NOTE_NAME, 0, vmcoreinfo_data,
         vmcoreinfo_size);
 final_note(buf);
}

void crash_save_vmcoreinfo(void)
{
 vmcoreinfo_append_str("CRASHTIME=%ld\n", get_seconds());
 update_vmcoreinfo_note();
}

void vmcoreinfo_append_str(const char *fmt, ...)
{
 va_list args;
 char buf[0x50];
 size_t r;

 va_start(args, fmt);
 r = vscnprintf(buf, sizeof(buf), fmt, args);
 va_end(args);

 r = min(r, vmcoreinfo_max_size - vmcoreinfo_size);

 memcpy(&vmcoreinfo_data[vmcoreinfo_size], buf, r);

 vmcoreinfo_size += r;
}





void __weak arch_crash_save_vmcoreinfo(void)
{}

unsigned long __weak paddr_vmcoreinfo_note(void)
{
 return __pa((unsigned long)(char *)&vmcoreinfo_note);
}

static int __init crash_save_vmcoreinfo_init(void)
{
 VMCOREINFO_OSRELEASE(init_uts_ns.name.release);
 VMCOREINFO_PAGESIZE(PAGE_SIZE);

 VMCOREINFO_SYMBOL(init_uts_ns);
 VMCOREINFO_SYMBOL(node_online_map);
 VMCOREINFO_SYMBOL(swapper_pg_dir);
 VMCOREINFO_SYMBOL(_stext);
 VMCOREINFO_SYMBOL(vmap_area_list);

 VMCOREINFO_SYMBOL(mem_map);
 VMCOREINFO_SYMBOL(contig_page_data);
 VMCOREINFO_SYMBOL(mem_section);
 VMCOREINFO_LENGTH(mem_section, NR_SECTION_ROOTS);
 VMCOREINFO_STRUCT_SIZE(mem_section);
 VMCOREINFO_OFFSET(mem_section, section_mem_map);
 VMCOREINFO_STRUCT_SIZE(page);
 VMCOREINFO_STRUCT_SIZE(pglist_data);
 VMCOREINFO_STRUCT_SIZE(zone);
 VMCOREINFO_STRUCT_SIZE(free_area);
 VMCOREINFO_STRUCT_SIZE(list_head);
 VMCOREINFO_SIZE(nodemask_t);
 VMCOREINFO_OFFSET(page, flags);
 VMCOREINFO_OFFSET(page, _refcount);
 VMCOREINFO_OFFSET(page, mapping);
 VMCOREINFO_OFFSET(page, lru);
 VMCOREINFO_OFFSET(page, _mapcount);
 VMCOREINFO_OFFSET(page, private);
 VMCOREINFO_OFFSET(page, compound_dtor);
 VMCOREINFO_OFFSET(page, compound_order);
 VMCOREINFO_OFFSET(page, compound_head);
 VMCOREINFO_OFFSET(pglist_data, node_zones);
 VMCOREINFO_OFFSET(pglist_data, nr_zones);
 VMCOREINFO_OFFSET(pglist_data, node_mem_map);
 VMCOREINFO_OFFSET(pglist_data, node_start_pfn);
 VMCOREINFO_OFFSET(pglist_data, node_spanned_pages);
 VMCOREINFO_OFFSET(pglist_data, node_id);
 VMCOREINFO_OFFSET(zone, free_area);
 VMCOREINFO_OFFSET(zone, vm_stat);
 VMCOREINFO_OFFSET(zone, spanned_pages);
 VMCOREINFO_OFFSET(free_area, free_list);
 VMCOREINFO_OFFSET(list_head, next);
 VMCOREINFO_OFFSET(list_head, prev);
 VMCOREINFO_OFFSET(vmap_area, va_start);
 VMCOREINFO_OFFSET(vmap_area, list);
 VMCOREINFO_LENGTH(zone.free_area, MAX_ORDER);
 log_buf_kexec_setup();
 VMCOREINFO_LENGTH(free_area.free_list, MIGRATE_TYPES);
 VMCOREINFO_NUMBER(NR_FREE_PAGES);
 VMCOREINFO_NUMBER(PG_lru);
 VMCOREINFO_NUMBER(PG_private);
 VMCOREINFO_NUMBER(PG_swapcache);
 VMCOREINFO_NUMBER(PG_slab);
 VMCOREINFO_NUMBER(PG_hwpoison);
 VMCOREINFO_NUMBER(PG_head_mask);
 VMCOREINFO_NUMBER(PAGE_BUDDY_MAPCOUNT_VALUE);
 VMCOREINFO_NUMBER(KERNEL_IMAGE_SIZE);
 VMCOREINFO_NUMBER(HUGETLB_PAGE_DTOR);

 arch_crash_save_vmcoreinfo();
 update_vmcoreinfo_note();

 return 0;
}

subsys_initcall(crash_save_vmcoreinfo_init);





int kernel_kexec(void)
{
 int error = 0;

 if (!mutex_trylock(&kexec_mutex))
  return -EBUSY;
 if (!kexec_image) {
  error = -EINVAL;
  goto Unlock;
 }

 if (kexec_image->preserve_context) {
  lock_system_sleep();
  pm_prepare_console();
  error = freeze_processes();
  if (error) {
   error = -EBUSY;
   goto Restore_console;
  }
  suspend_console();
  error = dpm_suspend_start(PMSG_FREEZE);
  if (error)
   goto Resume_console;







  error = dpm_suspend_end(PMSG_FREEZE);
  if (error)
   goto Resume_devices;
  error = disable_nonboot_cpus();
  if (error)
   goto Enable_cpus;
  local_irq_disable();
  error = syscore_suspend();
  if (error)
   goto Enable_irqs;
 } else
 {
  kexec_in_progress = true;
  kernel_restart_prepare(NULL);
  migrate_to_reboot_cpu();







  cpu_hotplug_enable();
  pr_emerg("Starting new kernel\n");
  machine_shutdown();
 }

 machine_kexec(kexec_image);

 if (kexec_image->preserve_context) {
  syscore_resume();
 Enable_irqs:
  local_irq_enable();
 Enable_cpus:
  enable_nonboot_cpus();
  dpm_resume_start(PMSG_RESTORE);
 Resume_devices:
  dpm_resume_end(PMSG_RESTORE);
 Resume_console:
  resume_console();
  thaw_processes();
 Restore_console:
  pm_restore_console();
  unlock_system_sleep();
 }

 Unlock:
 mutex_unlock(&kexec_mutex);
 return error;
}
void __weak arch_kexec_protect_crashkres(void)
{}

void __weak arch_kexec_unprotect_crashkres(void)
{}






char __weak kexec_purgatory[0];
size_t __weak kexec_purgatory_size = 0;

static int kexec_calculate_store_digests(struct kimage *image);


int __weak arch_kexec_kernel_image_probe(struct kimage *image, void *buf,
      unsigned long buf_len)
{
 return -ENOEXEC;
}

void * __weak arch_kexec_kernel_image_load(struct kimage *image)
{
 return ERR_PTR(-ENOEXEC);
}

int __weak arch_kimage_file_post_load_cleanup(struct kimage *image)
{
 return -EINVAL;
}

int __weak arch_kexec_kernel_verify_sig(struct kimage *image, void *buf,
     unsigned long buf_len)
{
 return -EKEYREJECTED;
}


int __weak
arch_kexec_apply_relocations_add(const Elf_Ehdr *ehdr, Elf_Shdr *sechdrs,
     unsigned int relsec)
{
 pr_err("RELA relocation unsupported.\n");
 return -ENOEXEC;
}


int __weak
arch_kexec_apply_relocations(const Elf_Ehdr *ehdr, Elf_Shdr *sechdrs,
        unsigned int relsec)
{
 pr_err("REL relocation unsupported.\n");
 return -ENOEXEC;
}






void kimage_file_post_load_cleanup(struct kimage *image)
{
 struct purgatory_info *pi = &image->purgatory_info;

 vfree(image->kernel_buf);
 image->kernel_buf = NULL;

 vfree(image->initrd_buf);
 image->initrd_buf = NULL;

 kfree(image->cmdline_buf);
 image->cmdline_buf = NULL;

 vfree(pi->purgatory_buf);
 pi->purgatory_buf = NULL;

 vfree(pi->sechdrs);
 pi->sechdrs = NULL;


 arch_kimage_file_post_load_cleanup(image);






 kfree(image->image_loader_data);
 image->image_loader_data = NULL;
}





static int
kimage_file_prepare_segments(struct kimage *image, int kernel_fd, int initrd_fd,
        const char __user *cmdline_ptr,
        unsigned long cmdline_len, unsigned flags)
{
 int ret = 0;
 void *ldata;
 loff_t size;

 ret = kernel_read_file_from_fd(kernel_fd, &image->kernel_buf,
           &size, INT_MAX, READING_KEXEC_IMAGE);
 if (ret)
  return ret;
 image->kernel_buf_len = size;


 ret = arch_kexec_kernel_image_probe(image, image->kernel_buf,
         image->kernel_buf_len);
 if (ret)
  goto out;

 ret = arch_kexec_kernel_verify_sig(image, image->kernel_buf,
        image->kernel_buf_len);
 if (ret) {
  pr_debug("kernel signature verification failed.\n");
  goto out;
 }
 pr_debug("kernel signature verification successful.\n");

 if (!(flags & KEXEC_FILE_NO_INITRAMFS)) {
  ret = kernel_read_file_from_fd(initrd_fd, &image->initrd_buf,
            &size, INT_MAX,
            READING_KEXEC_INITRAMFS);
  if (ret)
   goto out;
  image->initrd_buf_len = size;
 }

 if (cmdline_len) {
  image->cmdline_buf = kzalloc(cmdline_len, GFP_KERNEL);
  if (!image->cmdline_buf) {
   ret = -ENOMEM;
   goto out;
  }

  ret = copy_from_user(image->cmdline_buf, cmdline_ptr,
         cmdline_len);
  if (ret) {
   ret = -EFAULT;
   goto out;
  }

  image->cmdline_buf_len = cmdline_len;


  if (image->cmdline_buf[cmdline_len - 1] != '\0') {
   ret = -EINVAL;
   goto out;
  }
 }


 ldata = arch_kexec_kernel_image_load(image);

 if (IS_ERR(ldata)) {
  ret = PTR_ERR(ldata);
  goto out;
 }

 image->image_loader_data = ldata;
out:

 if (ret)
  kimage_file_post_load_cleanup(image);
 return ret;
}

static int
kimage_file_alloc_init(struct kimage **rimage, int kernel_fd,
         int initrd_fd, const char __user *cmdline_ptr,
         unsigned long cmdline_len, unsigned long flags)
{
 int ret;
 struct kimage *image;
 bool kexec_on_panic = flags & KEXEC_FILE_ON_CRASH;

 image = do_kimage_alloc_init();
 if (!image)
  return -ENOMEM;

 image->file_mode = 1;

 if (kexec_on_panic) {

  image->control_page = crashk_res.start;
  image->type = KEXEC_TYPE_CRASH;
 }

 ret = kimage_file_prepare_segments(image, kernel_fd, initrd_fd,
        cmdline_ptr, cmdline_len, flags);
 if (ret)
  goto out_free_image;

 ret = sanity_check_segment_list(image);
 if (ret)
  goto out_free_post_load_bufs;

 ret = -ENOMEM;
 image->control_code_page = kimage_alloc_control_pages(image,
        get_order(KEXEC_CONTROL_PAGE_SIZE));
 if (!image->control_code_page) {
  pr_err("Could not allocate control_code_buffer\n");
  goto out_free_post_load_bufs;
 }

 if (!kexec_on_panic) {
  image->swap_page = kimage_alloc_control_pages(image, 0);
  if (!image->swap_page) {
   pr_err("Could not allocate swap buffer\n");
   goto out_free_control_pages;
  }
 }

 *rimage = image;
 return 0;
out_free_control_pages:
 kimage_free_page_list(&image->control_pages);
out_free_post_load_bufs:
 kimage_file_post_load_cleanup(image);
out_free_image:
 kfree(image);
 return ret;
}

SYSCALL_DEFINE5(kexec_file_load, int, kernel_fd, int, initrd_fd,
  unsigned long, cmdline_len, const char __user *, cmdline_ptr,
  unsigned long, flags)
{
 int ret = 0, i;
 struct kimage **dest_image, *image;


 if (!capable(CAP_SYS_BOOT) || kexec_load_disabled)
  return -EPERM;


 if (flags != (flags & KEXEC_FILE_FLAGS))
  return -EINVAL;

 image = NULL;

 if (!mutex_trylock(&kexec_mutex))
  return -EBUSY;

 dest_image = &kexec_image;
 if (flags & KEXEC_FILE_ON_CRASH) {
  dest_image = &kexec_crash_image;
  if (kexec_crash_image)
   arch_kexec_unprotect_crashkres();
 }

 if (flags & KEXEC_FILE_UNLOAD)
  goto exchange;






 if (flags & KEXEC_FILE_ON_CRASH)
  kimage_free(xchg(&kexec_crash_image, NULL));

 ret = kimage_file_alloc_init(&image, kernel_fd, initrd_fd, cmdline_ptr,
         cmdline_len, flags);
 if (ret)
  goto out;

 ret = machine_kexec_prepare(image);
 if (ret)
  goto out;

 ret = kexec_calculate_store_digests(image);
 if (ret)
  goto out;

 for (i = 0; i < image->nr_segments; i++) {
  struct kexec_segment *ksegment;

  ksegment = &image->segment[i];
  pr_debug("Loading segment %d: buf=0x%p bufsz=0x%zx mem=0x%lx memsz=0x%zx\n",
    i, ksegment->buf, ksegment->bufsz, ksegment->mem,
    ksegment->memsz);

  ret = kimage_load_segment(image, &image->segment[i]);
  if (ret)
   goto out;
 }

 kimage_terminate(image);





 kimage_file_post_load_cleanup(image);
exchange:
 image = xchg(dest_image, image);
out:
 if ((flags & KEXEC_FILE_ON_CRASH) && kexec_crash_image)
  arch_kexec_protect_crashkres();

 mutex_unlock(&kexec_mutex);
 kimage_free(image);
 return ret;
}

static int locate_mem_hole_top_down(unsigned long start, unsigned long end,
        struct kexec_buf *kbuf)
{
 struct kimage *image = kbuf->image;
 unsigned long temp_start, temp_end;

 temp_end = min(end, kbuf->buf_max);
 temp_start = temp_end - kbuf->memsz;

 do {

  temp_start = temp_start & (~(kbuf->buf_align - 1));

  if (temp_start < start || temp_start < kbuf->buf_min)
   return 0;

  temp_end = temp_start + kbuf->memsz - 1;





  if (kimage_is_destination_range(image, temp_start, temp_end)) {
   temp_start = temp_start - PAGE_SIZE;
   continue;
  }


  break;
 } while (1);


 kbuf->mem = temp_start;


 return 1;
}

static int locate_mem_hole_bottom_up(unsigned long start, unsigned long end,
         struct kexec_buf *kbuf)
{
 struct kimage *image = kbuf->image;
 unsigned long temp_start, temp_end;

 temp_start = max(start, kbuf->buf_min);

 do {
  temp_start = ALIGN(temp_start, kbuf->buf_align);
  temp_end = temp_start + kbuf->memsz - 1;

  if (temp_end > end || temp_end > kbuf->buf_max)
   return 0;




  if (kimage_is_destination_range(image, temp_start, temp_end)) {
   temp_start = temp_start + PAGE_SIZE;
   continue;
  }


  break;
 } while (1);


 kbuf->mem = temp_start;


 return 1;
}

static int locate_mem_hole_callback(u64 start, u64 end, void *arg)
{
 struct kexec_buf *kbuf = (struct kexec_buf *)arg;
 unsigned long sz = end - start + 1;


 if (sz < kbuf->memsz)
  return 0;

 if (end < kbuf->buf_min || start > kbuf->buf_max)
  return 0;





 if (kbuf->top_down)
  return locate_mem_hole_top_down(start, end, kbuf);
 return locate_mem_hole_bottom_up(start, end, kbuf);
}





int kexec_add_buffer(struct kimage *image, char *buffer, unsigned long bufsz,
       unsigned long memsz, unsigned long buf_align,
       unsigned long buf_min, unsigned long buf_max,
       bool top_down, unsigned long *load_addr)
{

 struct kexec_segment *ksegment;
 struct kexec_buf buf, *kbuf;
 int ret;


 if (!image->file_mode)
  return -EINVAL;

 if (image->nr_segments >= KEXEC_SEGMENT_MAX)
  return -EINVAL;
 if (!list_empty(&image->control_pages)) {
  WARN_ON(1);
  return -EINVAL;
 }

 memset(&buf, 0, sizeof(struct kexec_buf));
 kbuf = &buf;
 kbuf->image = image;
 kbuf->buffer = buffer;
 kbuf->bufsz = bufsz;

 kbuf->memsz = ALIGN(memsz, PAGE_SIZE);
 kbuf->buf_align = max(buf_align, PAGE_SIZE);
 kbuf->buf_min = buf_min;
 kbuf->buf_max = buf_max;
 kbuf->top_down = top_down;


 if (image->type == KEXEC_TYPE_CRASH)
  ret = walk_iomem_res_desc(crashk_res.desc,
    IORESOURCE_SYSTEM_RAM | IORESOURCE_BUSY,
    crashk_res.start, crashk_res.end, kbuf,
    locate_mem_hole_callback);
 else
  ret = walk_system_ram_res(0, -1, kbuf,
       locate_mem_hole_callback);
 if (ret != 1) {

  return -EADDRNOTAVAIL;
 }


 ksegment = &image->segment[image->nr_segments];
 ksegment->kbuf = kbuf->buffer;
 ksegment->bufsz = kbuf->bufsz;
 ksegment->mem = kbuf->mem;
 ksegment->memsz = kbuf->memsz;
 image->nr_segments++;
 *load_addr = ksegment->mem;
 return 0;
}


static int kexec_calculate_store_digests(struct kimage *image)
{
 struct crypto_shash *tfm;
 struct shash_desc *desc;
 int ret = 0, i, j, zero_buf_sz, sha_region_sz;
 size_t desc_size, nullsz;
 char *digest;
 void *zero_buf;
 struct kexec_sha_region *sha_regions;
 struct purgatory_info *pi = &image->purgatory_info;

 zero_buf = __va(page_to_pfn(ZERO_PAGE(0)) << PAGE_SHIFT);
 zero_buf_sz = PAGE_SIZE;

 tfm = crypto_alloc_shash("sha256", 0, 0);
 if (IS_ERR(tfm)) {
  ret = PTR_ERR(tfm);
  goto out;
 }

 desc_size = crypto_shash_descsize(tfm) + sizeof(*desc);
 desc = kzalloc(desc_size, GFP_KERNEL);
 if (!desc) {
  ret = -ENOMEM;
  goto out_free_tfm;
 }

 sha_region_sz = KEXEC_SEGMENT_MAX * sizeof(struct kexec_sha_region);
 sha_regions = vzalloc(sha_region_sz);
 if (!sha_regions)
  goto out_free_desc;

 desc->tfm = tfm;
 desc->flags = 0;

 ret = crypto_shash_init(desc);
 if (ret < 0)
  goto out_free_sha_regions;

 digest = kzalloc(SHA256_DIGEST_SIZE, GFP_KERNEL);
 if (!digest) {
  ret = -ENOMEM;
  goto out_free_sha_regions;
 }

 for (j = i = 0; i < image->nr_segments; i++) {
  struct kexec_segment *ksegment;

  ksegment = &image->segment[i];




  if (ksegment->kbuf == pi->purgatory_buf)
   continue;

  ret = crypto_shash_update(desc, ksegment->kbuf,
       ksegment->bufsz);
  if (ret)
   break;





  nullsz = ksegment->memsz - ksegment->bufsz;
  while (nullsz) {
   unsigned long bytes = nullsz;

   if (bytes > zero_buf_sz)
    bytes = zero_buf_sz;
   ret = crypto_shash_update(desc, zero_buf, bytes);
   if (ret)
    break;
   nullsz -= bytes;
  }

  if (ret)
   break;

  sha_regions[j].start = ksegment->mem;
  sha_regions[j].len = ksegment->memsz;
  j++;
 }

 if (!ret) {
  ret = crypto_shash_final(desc, digest);
  if (ret)
   goto out_free_digest;
  ret = kexec_purgatory_get_set_symbol(image, "sha_regions",
      sha_regions, sha_region_sz, 0);
  if (ret)
   goto out_free_digest;

  ret = kexec_purgatory_get_set_symbol(image, "sha256_digest",
      digest, SHA256_DIGEST_SIZE, 0);
  if (ret)
   goto out_free_digest;
 }

out_free_digest:
 kfree(digest);
out_free_sha_regions:
 vfree(sha_regions);
out_free_desc:
 kfree(desc);
out_free_tfm:
 kfree(tfm);
out:
 return ret;
}


static int __kexec_load_purgatory(struct kimage *image, unsigned long min,
      unsigned long max, int top_down)
{
 struct purgatory_info *pi = &image->purgatory_info;
 unsigned long align, buf_align, bss_align, buf_sz, bss_sz, bss_pad;
 unsigned long memsz, entry, load_addr, curr_load_addr, bss_addr, offset;
 unsigned char *buf_addr, *src;
 int i, ret = 0, entry_sidx = -1;
 const Elf_Shdr *sechdrs_c;
 Elf_Shdr *sechdrs = NULL;
 void *purgatory_buf = NULL;





 sechdrs_c = (void *)pi->ehdr + pi->ehdr->e_shoff;
 sechdrs = vzalloc(pi->ehdr->e_shnum * sizeof(Elf_Shdr));
 if (!sechdrs)
  return -ENOMEM;

 memcpy(sechdrs, sechdrs_c, pi->ehdr->e_shnum * sizeof(Elf_Shdr));
 for (i = 0; i < pi->ehdr->e_shnum; i++) {
  if (sechdrs[i].sh_type == SHT_NOBITS)
   continue;

  sechdrs[i].sh_offset = (unsigned long)pi->ehdr +
      sechdrs[i].sh_offset;
 }





 entry = pi->ehdr->e_entry;
 for (i = 0; i < pi->ehdr->e_shnum; i++) {
  if (!(sechdrs[i].sh_flags & SHF_ALLOC))
   continue;

  if (!(sechdrs[i].sh_flags & SHF_EXECINSTR))
   continue;


  if (sechdrs[i].sh_addr <= pi->ehdr->e_entry &&
      ((sechdrs[i].sh_addr + sechdrs[i].sh_size) >
       pi->ehdr->e_entry)) {
   entry_sidx = i;
   entry -= sechdrs[i].sh_addr;
   break;
  }
 }


 buf_align = 1;
 bss_align = 1;
 buf_sz = 0;
 bss_sz = 0;

 for (i = 0; i < pi->ehdr->e_shnum; i++) {
  if (!(sechdrs[i].sh_flags & SHF_ALLOC))
   continue;

  align = sechdrs[i].sh_addralign;
  if (sechdrs[i].sh_type != SHT_NOBITS) {
   if (buf_align < align)
    buf_align = align;
   buf_sz = ALIGN(buf_sz, align);
   buf_sz += sechdrs[i].sh_size;
  } else {

   if (bss_align < align)
    bss_align = align;
   bss_sz = ALIGN(bss_sz, align);
   bss_sz += sechdrs[i].sh_size;
  }
 }


 bss_pad = 0;
 if (buf_sz & (bss_align - 1))
  bss_pad = bss_align - (buf_sz & (bss_align - 1));

 memsz = buf_sz + bss_pad + bss_sz;


 purgatory_buf = vzalloc(buf_sz);
 if (!purgatory_buf) {
  ret = -ENOMEM;
  goto out;
 }

 if (buf_align < bss_align)
  buf_align = bss_align;


 ret = kexec_add_buffer(image, purgatory_buf, buf_sz, memsz,
    buf_align, min, max, top_down,
    &pi->purgatory_load_addr);
 if (ret)
  goto out;


 buf_addr = purgatory_buf;
 load_addr = curr_load_addr = pi->purgatory_load_addr;
 bss_addr = load_addr + buf_sz + bss_pad;

 for (i = 0; i < pi->ehdr->e_shnum; i++) {
  if (!(sechdrs[i].sh_flags & SHF_ALLOC))
   continue;

  align = sechdrs[i].sh_addralign;
  if (sechdrs[i].sh_type != SHT_NOBITS) {
   curr_load_addr = ALIGN(curr_load_addr, align);
   offset = curr_load_addr - load_addr;

   src = (char *) sechdrs[i].sh_offset;
   memcpy(buf_addr + offset, src, sechdrs[i].sh_size);


   sechdrs[i].sh_addr = curr_load_addr;





   sechdrs[i].sh_offset = (unsigned long)(buf_addr + offset);


   curr_load_addr += sechdrs[i].sh_size;
  } else {
   bss_addr = ALIGN(bss_addr, align);
   sechdrs[i].sh_addr = bss_addr;
   bss_addr += sechdrs[i].sh_size;
  }
 }


 if (entry_sidx >= 0)
  entry += sechdrs[entry_sidx].sh_addr;


 image->start = entry;


 pi->sechdrs = sechdrs;





 pi->purgatory_buf = purgatory_buf;
 return ret;
out:
 vfree(sechdrs);
 vfree(purgatory_buf);
 return ret;
}

static int kexec_apply_relocations(struct kimage *image)
{
 int i, ret;
 struct purgatory_info *pi = &image->purgatory_info;
 Elf_Shdr *sechdrs = pi->sechdrs;


 for (i = 0; i < pi->ehdr->e_shnum; i++) {
  Elf_Shdr *section, *symtab;

  if (sechdrs[i].sh_type != SHT_RELA &&
      sechdrs[i].sh_type != SHT_REL)
   continue;







  if (sechdrs[i].sh_info >= pi->ehdr->e_shnum ||
      sechdrs[i].sh_link >= pi->ehdr->e_shnum)
   return -ENOEXEC;

  section = &sechdrs[sechdrs[i].sh_info];
  symtab = &sechdrs[sechdrs[i].sh_link];

  if (!(section->sh_flags & SHF_ALLOC))
   continue;





  if (symtab->sh_link >= pi->ehdr->e_shnum)

   continue;





  if (sechdrs[i].sh_type == SHT_RELA)
   ret = arch_kexec_apply_relocations_add(pi->ehdr,
              sechdrs, i);
  else if (sechdrs[i].sh_type == SHT_REL)
   ret = arch_kexec_apply_relocations(pi->ehdr,
          sechdrs, i);
  if (ret)
   return ret;
 }

 return 0;
}


int kexec_load_purgatory(struct kimage *image, unsigned long min,
    unsigned long max, int top_down,
    unsigned long *load_addr)
{
 struct purgatory_info *pi = &image->purgatory_info;
 int ret;

 if (kexec_purgatory_size <= 0)
  return -EINVAL;

 if (kexec_purgatory_size < sizeof(Elf_Ehdr))
  return -ENOEXEC;

 pi->ehdr = (Elf_Ehdr *)kexec_purgatory;

 if (memcmp(pi->ehdr->e_ident, ELFMAG, SELFMAG) != 0
     || pi->ehdr->e_type != ET_REL
     || !elf_check_arch(pi->ehdr)
     || pi->ehdr->e_shentsize != sizeof(Elf_Shdr))
  return -ENOEXEC;

 if (pi->ehdr->e_shoff >= kexec_purgatory_size
     || (pi->ehdr->e_shnum * sizeof(Elf_Shdr) >
     kexec_purgatory_size - pi->ehdr->e_shoff))
  return -ENOEXEC;

 ret = __kexec_load_purgatory(image, min, max, top_down);
 if (ret)
  return ret;

 ret = kexec_apply_relocations(image);
 if (ret)
  goto out;

 *load_addr = pi->purgatory_load_addr;
 return 0;
out:
 vfree(pi->sechdrs);
 vfree(pi->purgatory_buf);
 return ret;
}

static Elf_Sym *kexec_purgatory_find_symbol(struct purgatory_info *pi,
         const char *name)
{
 Elf_Sym *syms;
 Elf_Shdr *sechdrs;
 Elf_Ehdr *ehdr;
 int i, k;
 const char *strtab;

 if (!pi->sechdrs || !pi->ehdr)
  return NULL;

 sechdrs = pi->sechdrs;
 ehdr = pi->ehdr;

 for (i = 0; i < ehdr->e_shnum; i++) {
  if (sechdrs[i].sh_type != SHT_SYMTAB)
   continue;

  if (sechdrs[i].sh_link >= ehdr->e_shnum)

   continue;
  strtab = (char *)sechdrs[sechdrs[i].sh_link].sh_offset;
  syms = (Elf_Sym *)sechdrs[i].sh_offset;


  for (k = 0; k < sechdrs[i].sh_size/sizeof(Elf_Sym); k++) {
   if (ELF_ST_BIND(syms[k].st_info) != STB_GLOBAL)
    continue;

   if (strcmp(strtab + syms[k].st_name, name) != 0)
    continue;

   if (syms[k].st_shndx == SHN_UNDEF ||
       syms[k].st_shndx >= ehdr->e_shnum) {
    pr_debug("Symbol: %s has bad section index %d.\n",
      name, syms[k].st_shndx);
    return NULL;
   }


   return &syms[k];
  }
 }

 return NULL;
}

void *kexec_purgatory_get_symbol_addr(struct kimage *image, const char *name)
{
 struct purgatory_info *pi = &image->purgatory_info;
 Elf_Sym *sym;
 Elf_Shdr *sechdr;

 sym = kexec_purgatory_find_symbol(pi, name);
 if (!sym)
  return ERR_PTR(-EINVAL);

 sechdr = &pi->sechdrs[sym->st_shndx];





 return (void *)(sechdr->sh_addr + sym->st_value);
}





int kexec_purgatory_get_set_symbol(struct kimage *image, const char *name,
       void *buf, unsigned int size, bool get_value)
{
 Elf_Sym *sym;
 Elf_Shdr *sechdrs;
 struct purgatory_info *pi = &image->purgatory_info;
 char *sym_buf;

 sym = kexec_purgatory_find_symbol(pi, name);
 if (!sym)
  return -EINVAL;

 if (sym->st_size != size) {
  pr_err("symbol %s size mismatch: expected %lu actual %u\n",
         name, (unsigned long)sym->st_size, size);
  return -EINVAL;
 }

 sechdrs = pi->sechdrs;

 if (sechdrs[sym->st_shndx].sh_type == SHT_NOBITS) {
  pr_err("symbol %s is in a bss section. Cannot %s\n", name,
         get_value ? "get" : "set");
  return -EINVAL;
 }

 sym_buf = (unsigned char *)sechdrs[sym->st_shndx].sh_offset +
     sym->st_value;

 if (get_value)
  memcpy((void *)buf, sym_buf, size);
 else
  memcpy((void *)sym_buf, buf, size);

 return 0;
}


extern int max_threads;


static kernel_cap_t usermodehelper_bset = CAP_FULL_SET;
static kernel_cap_t usermodehelper_inheritable = CAP_FULL_SET;
static DEFINE_SPINLOCK(umh_sysctl_lock);
static DECLARE_RWSEM(umhelper_sem);





char modprobe_path[KMOD_PATH_LEN] = "/sbin/modprobe";

static void free_modprobe_argv(struct subprocess_info *info)
{
 kfree(info->argv[3]);
 kfree(info->argv);
}

static int call_modprobe(char *module_name, int wait)
{
 struct subprocess_info *info;
 static char *envp[] = {
  "HOME=/",
  "TERM=linux",
  "PATH=/sbin:/usr/sbin:/bin:/usr/bin",
  NULL
 };

 char **argv = kmalloc(sizeof(char *[5]), GFP_KERNEL);
 if (!argv)
  goto out;

 module_name = kstrdup(module_name, GFP_KERNEL);
 if (!module_name)
  goto free_argv;

 argv[0] = modprobe_path;
 argv[1] = "-q";
 argv[2] = "--";
 argv[3] = module_name;
 argv[4] = NULL;

 info = call_usermodehelper_setup(modprobe_path, argv, envp, GFP_KERNEL,
      NULL, free_modprobe_argv, NULL);
 if (!info)
  goto free_module_name;

 return call_usermodehelper_exec(info, wait | UMH_KILLABLE);

free_module_name:
 kfree(module_name);
free_argv:
 kfree(argv);
out:
 return -ENOMEM;
}
int __request_module(bool wait, const char *fmt, ...)
{
 va_list args;
 char module_name[MODULE_NAME_LEN];
 unsigned int max_modprobes;
 int ret;
 static atomic_t kmod_concurrent = ATOMIC_INIT(0);
 static int kmod_loop_msg;







 WARN_ON_ONCE(wait && current_is_async());

 if (!modprobe_path[0])
  return 0;

 va_start(args, fmt);
 ret = vsnprintf(module_name, MODULE_NAME_LEN, fmt, args);
 va_end(args);
 if (ret >= MODULE_NAME_LEN)
  return -ENAMETOOLONG;

 ret = security_kernel_module_request(module_name);
 if (ret)
  return ret;
 max_modprobes = min(max_threads/2, MAX_KMOD_CONCURRENT);
 atomic_inc(&kmod_concurrent);
 if (atomic_read(&kmod_concurrent) > max_modprobes) {

  if (kmod_loop_msg < 5) {
   printk(KERN_ERR
          "request_module: runaway loop modprobe %s\n",
          module_name);
   kmod_loop_msg++;
  }
  atomic_dec(&kmod_concurrent);
  return -ENOMEM;
 }

 trace_module_request(module_name, wait, _RET_IP_);

 ret = call_modprobe(module_name, wait ? UMH_WAIT_PROC : UMH_WAIT_EXEC);

 atomic_dec(&kmod_concurrent);
 return ret;
}
EXPORT_SYMBOL(__request_module);

static void call_usermodehelper_freeinfo(struct subprocess_info *info)
{
 if (info->cleanup)
  (*info->cleanup)(info);
 kfree(info);
}

static void umh_complete(struct subprocess_info *sub_info)
{
 struct completion *comp = xchg(&sub_info->complete, NULL);





 if (comp)
  complete(comp);
 else
  call_usermodehelper_freeinfo(sub_info);
}




static int call_usermodehelper_exec_async(void *data)
{
 struct subprocess_info *sub_info = data;
 struct cred *new;
 int retval;

 spin_lock_irq(&current->sighand->siglock);
 flush_signal_handlers(current, 1);
 spin_unlock_irq(&current->sighand->siglock);





 set_user_nice(current, 0);

 retval = -ENOMEM;
 new = prepare_kernel_cred(current);
 if (!new)
  goto out;

 spin_lock(&umh_sysctl_lock);
 new->cap_bset = cap_intersect(usermodehelper_bset, new->cap_bset);
 new->cap_inheritable = cap_intersect(usermodehelper_inheritable,
          new->cap_inheritable);
 spin_unlock(&umh_sysctl_lock);

 if (sub_info->init) {
  retval = sub_info->init(sub_info, new);
  if (retval) {
   abort_creds(new);
   goto out;
  }
 }

 commit_creds(new);

 retval = do_execve(getname_kernel(sub_info->path),
      (const char __user *const __user *)sub_info->argv,
      (const char __user *const __user *)sub_info->envp);
out:
 sub_info->retval = retval;




 if (!(sub_info->wait & UMH_WAIT_PROC))
  umh_complete(sub_info);
 if (!retval)
  return 0;
 do_exit(0);
}


static void call_usermodehelper_exec_sync(struct subprocess_info *sub_info)
{
 pid_t pid;


 kernel_sigaction(SIGCHLD, SIG_DFL);
 pid = kernel_thread(call_usermodehelper_exec_async, sub_info, SIGCHLD);
 if (pid < 0) {
  sub_info->retval = pid;
 } else {
  int ret = -ECHILD;
  sys_wait4(pid, (int __user *)&ret, 0, NULL);






  if (ret)
   sub_info->retval = ret;
 }


 kernel_sigaction(SIGCHLD, SIG_IGN);

 umh_complete(sub_info);
}
static void call_usermodehelper_exec_work(struct work_struct *work)
{
 struct subprocess_info *sub_info =
  container_of(work, struct subprocess_info, work);

 if (sub_info->wait & UMH_WAIT_PROC) {
  call_usermodehelper_exec_sync(sub_info);
 } else {
  pid_t pid;





  pid = kernel_thread(call_usermodehelper_exec_async, sub_info,
        CLONE_PARENT | SIGCHLD);
  if (pid < 0) {
   sub_info->retval = pid;
   umh_complete(sub_info);
  }
 }
}







static enum umh_disable_depth usermodehelper_disabled = UMH_DISABLED;


static atomic_t running_helpers = ATOMIC_INIT(0);





static DECLARE_WAIT_QUEUE_HEAD(running_helpers_waitq);





static DECLARE_WAIT_QUEUE_HEAD(usermodehelper_disabled_waitq);






int usermodehelper_read_trylock(void)
{
 DEFINE_WAIT(wait);
 int ret = 0;

 down_read(&umhelper_sem);
 for (;;) {
  prepare_to_wait(&usermodehelper_disabled_waitq, &wait,
    TASK_INTERRUPTIBLE);
  if (!usermodehelper_disabled)
   break;

  if (usermodehelper_disabled == UMH_DISABLED)
   ret = -EAGAIN;

  up_read(&umhelper_sem);

  if (ret)
   break;

  schedule();
  try_to_freeze();

  down_read(&umhelper_sem);
 }
 finish_wait(&usermodehelper_disabled_waitq, &wait);
 return ret;
}
EXPORT_SYMBOL_GPL(usermodehelper_read_trylock);

long usermodehelper_read_lock_wait(long timeout)
{
 DEFINE_WAIT(wait);

 if (timeout < 0)
  return -EINVAL;

 down_read(&umhelper_sem);
 for (;;) {
  prepare_to_wait(&usermodehelper_disabled_waitq, &wait,
    TASK_UNINTERRUPTIBLE);
  if (!usermodehelper_disabled)
   break;

  up_read(&umhelper_sem);

  timeout = schedule_timeout(timeout);
  if (!timeout)
   break;

  down_read(&umhelper_sem);
 }
 finish_wait(&usermodehelper_disabled_waitq, &wait);
 return timeout;
}
EXPORT_SYMBOL_GPL(usermodehelper_read_lock_wait);

void usermodehelper_read_unlock(void)
{
 up_read(&umhelper_sem);
}
EXPORT_SYMBOL_GPL(usermodehelper_read_unlock);
void __usermodehelper_set_disable_depth(enum umh_disable_depth depth)
{
 down_write(&umhelper_sem);
 usermodehelper_disabled = depth;
 wake_up(&usermodehelper_disabled_waitq);
 up_write(&umhelper_sem);
}







int __usermodehelper_disable(enum umh_disable_depth depth)
{
 long retval;

 if (!depth)
  return -EINVAL;

 down_write(&umhelper_sem);
 usermodehelper_disabled = depth;
 up_write(&umhelper_sem);







 retval = wait_event_timeout(running_helpers_waitq,
     atomic_read(&running_helpers) == 0,
     RUNNING_HELPERS_TIMEOUT);
 if (retval)
  return 0;

 __usermodehelper_set_disable_depth(UMH_ENABLED);
 return -EAGAIN;
}

static void helper_lock(void)
{
 atomic_inc(&running_helpers);
 smp_mb__after_atomic();
}

static void helper_unlock(void)
{
 if (atomic_dec_and_test(&running_helpers))
  wake_up(&running_helpers_waitq);
}
struct subprocess_info *call_usermodehelper_setup(char *path, char **argv,
  char **envp, gfp_t gfp_mask,
  int (*init)(struct subprocess_info *info, struct cred *new),
  void (*cleanup)(struct subprocess_info *info),
  void *data)
{
 struct subprocess_info *sub_info;
 sub_info = kzalloc(sizeof(struct subprocess_info), gfp_mask);
 if (!sub_info)
  goto out;

 INIT_WORK(&sub_info->work, call_usermodehelper_exec_work);
 sub_info->path = path;
 sub_info->argv = argv;
 sub_info->envp = envp;

 sub_info->cleanup = cleanup;
 sub_info->init = init;
 sub_info->data = data;
  out:
 return sub_info;
}
EXPORT_SYMBOL(call_usermodehelper_setup);
int call_usermodehelper_exec(struct subprocess_info *sub_info, int wait)
{
 DECLARE_COMPLETION_ONSTACK(done);
 int retval = 0;

 if (!sub_info->path) {
  call_usermodehelper_freeinfo(sub_info);
  return -EINVAL;
 }
 helper_lock();
 if (usermodehelper_disabled) {
  retval = -EBUSY;
  goto out;
 }





 sub_info->complete = (wait == UMH_NO_WAIT) ? NULL : &done;
 sub_info->wait = wait;

 queue_work(system_unbound_wq, &sub_info->work);
 if (wait == UMH_NO_WAIT)
  goto unlock;

 if (wait & UMH_KILLABLE) {
  retval = wait_for_completion_killable(&done);
  if (!retval)
   goto wait_done;


  if (xchg(&sub_info->complete, NULL))
   goto unlock;

 }

 wait_for_completion(&done);
wait_done:
 retval = sub_info->retval;
out:
 call_usermodehelper_freeinfo(sub_info);
unlock:
 helper_unlock();
 return retval;
}
EXPORT_SYMBOL(call_usermodehelper_exec);
int call_usermodehelper(char *path, char **argv, char **envp, int wait)
{
 struct subprocess_info *info;
 gfp_t gfp_mask = (wait == UMH_NO_WAIT) ? GFP_ATOMIC : GFP_KERNEL;

 info = call_usermodehelper_setup(path, argv, envp, gfp_mask,
      NULL, NULL, NULL);
 if (info == NULL)
  return -ENOMEM;

 return call_usermodehelper_exec(info, wait);
}
EXPORT_SYMBOL(call_usermodehelper);

static int proc_cap_handler(struct ctl_table *table, int write,
    void __user *buffer, size_t *lenp, loff_t *ppos)
{
 struct ctl_table t;
 unsigned long cap_array[_KERNEL_CAPABILITY_U32S];
 kernel_cap_t new_cap;
 int err, i;

 if (write && (!capable(CAP_SETPCAP) ||
        !capable(CAP_SYS_MODULE)))
  return -EPERM;





 spin_lock(&umh_sysctl_lock);
 for (i = 0; i < _KERNEL_CAPABILITY_U32S; i++) {
  if (table->data == CAP_BSET)
   cap_array[i] = usermodehelper_bset.cap[i];
  else if (table->data == CAP_PI)
   cap_array[i] = usermodehelper_inheritable.cap[i];
  else
   BUG();
 }
 spin_unlock(&umh_sysctl_lock);

 t = *table;
 t.data = &cap_array;





 err = proc_doulongvec_minmax(&t, write, buffer, lenp, ppos);
 if (err < 0)
  return err;





 for (i = 0; i < _KERNEL_CAPABILITY_U32S; i++)
  new_cap.cap[i] = cap_array[i];




 spin_lock(&umh_sysctl_lock);
 if (write) {
  if (table->data == CAP_BSET)
   usermodehelper_bset = cap_intersect(usermodehelper_bset, new_cap);
  if (table->data == CAP_PI)
   usermodehelper_inheritable = cap_intersect(usermodehelper_inheritable, new_cap);
 }
 spin_unlock(&umh_sysctl_lock);

 return 0;
}

struct ctl_table usermodehelper_table[] = {
 {
  .procname = "bset",
  .data = CAP_BSET,
  .maxlen = _KERNEL_CAPABILITY_U32S * sizeof(unsigned long),
  .mode = 0600,
  .proc_handler = proc_cap_handler,
 },
 {
  .procname = "inheritable",
  .data = CAP_PI,
  .maxlen = _KERNEL_CAPABILITY_U32S * sizeof(unsigned long),
  .mode = 0600,
  .proc_handler = proc_cap_handler,
 },
 { }
};








 addr = ((kprobe_opcode_t *)(kallsyms_lookup_name(name)))

static int kprobes_initialized;
static struct hlist_head kprobe_table[KPROBE_TABLE_SIZE];
static struct hlist_head kretprobe_inst_table[KPROBE_TABLE_SIZE];


static bool kprobes_all_disarmed;


static DEFINE_MUTEX(kprobe_mutex);
static DEFINE_PER_CPU(struct kprobe *, kprobe_instance) = NULL;
static struct {
 raw_spinlock_t lock ____cacheline_aligned_in_smp;
} kretprobe_table_locks[KPROBE_TABLE_SIZE];

static raw_spinlock_t *kretprobe_table_lock_ptr(unsigned long hash)
{
 return &(kretprobe_table_locks[hash].lock);
}


static LIST_HEAD(kprobe_blacklist);







struct kprobe_insn_page {
 struct list_head list;
 kprobe_opcode_t *insns;
 struct kprobe_insn_cache *cache;
 int nused;
 int ngarbage;
 char slot_used[];
};

 (offsetof(struct kprobe_insn_page, slot_used) + \
  (sizeof(char) * (slots)))

static int slots_per_page(struct kprobe_insn_cache *c)
{
 return PAGE_SIZE/(c->insn_size * sizeof(kprobe_opcode_t));
}

enum kprobe_slot_state {
 SLOT_CLEAN = 0,
 SLOT_DIRTY = 1,
 SLOT_USED = 2,
};

static void *alloc_insn_page(void)
{
 return module_alloc(PAGE_SIZE);
}

static void free_insn_page(void *page)
{
 module_memfree(page);
}

struct kprobe_insn_cache kprobe_insn_slots = {
 .mutex = __MUTEX_INITIALIZER(kprobe_insn_slots.mutex),
 .alloc = alloc_insn_page,
 .free = free_insn_page,
 .pages = LIST_HEAD_INIT(kprobe_insn_slots.pages),
 .insn_size = MAX_INSN_SIZE,
 .nr_garbage = 0,
};
static int collect_garbage_slots(struct kprobe_insn_cache *c);





kprobe_opcode_t *__get_insn_slot(struct kprobe_insn_cache *c)
{
 struct kprobe_insn_page *kip;
 kprobe_opcode_t *slot = NULL;

 mutex_lock(&c->mutex);
 retry:
 list_for_each_entry(kip, &c->pages, list) {
  if (kip->nused < slots_per_page(c)) {
   int i;
   for (i = 0; i < slots_per_page(c); i++) {
    if (kip->slot_used[i] == SLOT_CLEAN) {
     kip->slot_used[i] = SLOT_USED;
     kip->nused++;
     slot = kip->insns + (i * c->insn_size);
     goto out;
    }
   }

   kip->nused = slots_per_page(c);
   WARN_ON(1);
  }
 }


 if (c->nr_garbage && collect_garbage_slots(c) == 0)
  goto retry;


 kip = kmalloc(KPROBE_INSN_PAGE_SIZE(slots_per_page(c)), GFP_KERNEL);
 if (!kip)
  goto out;






 kip->insns = c->alloc();
 if (!kip->insns) {
  kfree(kip);
  goto out;
 }
 INIT_LIST_HEAD(&kip->list);
 memset(kip->slot_used, SLOT_CLEAN, slots_per_page(c));
 kip->slot_used[0] = SLOT_USED;
 kip->nused = 1;
 kip->ngarbage = 0;
 kip->cache = c;
 list_add(&kip->list, &c->pages);
 slot = kip->insns;
out:
 mutex_unlock(&c->mutex);
 return slot;
}


static int collect_one_slot(struct kprobe_insn_page *kip, int idx)
{
 kip->slot_used[idx] = SLOT_CLEAN;
 kip->nused--;
 if (kip->nused == 0) {






  if (!list_is_singular(&kip->list)) {
   list_del(&kip->list);
   kip->cache->free(kip->insns);
   kfree(kip);
  }
  return 1;
 }
 return 0;
}

static int collect_garbage_slots(struct kprobe_insn_cache *c)
{
 struct kprobe_insn_page *kip, *next;


 synchronize_sched();

 list_for_each_entry_safe(kip, next, &c->pages, list) {
  int i;
  if (kip->ngarbage == 0)
   continue;
  kip->ngarbage = 0;
  for (i = 0; i < slots_per_page(c); i++) {
   if (kip->slot_used[i] == SLOT_DIRTY &&
       collect_one_slot(kip, i))
    break;
  }
 }
 c->nr_garbage = 0;
 return 0;
}

void __free_insn_slot(struct kprobe_insn_cache *c,
        kprobe_opcode_t *slot, int dirty)
{
 struct kprobe_insn_page *kip;

 mutex_lock(&c->mutex);
 list_for_each_entry(kip, &c->pages, list) {
  long idx = ((long)slot - (long)kip->insns) /
    (c->insn_size * sizeof(kprobe_opcode_t));
  if (idx >= 0 && idx < slots_per_page(c)) {
   WARN_ON(kip->slot_used[idx] != SLOT_USED);
   if (dirty) {
    kip->slot_used[idx] = SLOT_DIRTY;
    kip->ngarbage++;
    if (++c->nr_garbage > slots_per_page(c))
     collect_garbage_slots(c);
   } else
    collect_one_slot(kip, idx);
   goto out;
  }
 }

 WARN_ON(1);
out:
 mutex_unlock(&c->mutex);
}


struct kprobe_insn_cache kprobe_optinsn_slots = {
 .mutex = __MUTEX_INITIALIZER(kprobe_optinsn_slots.mutex),
 .alloc = alloc_insn_page,
 .free = free_insn_page,
 .pages = LIST_HEAD_INIT(kprobe_optinsn_slots.pages),

 .nr_garbage = 0,
};


static inline void set_kprobe_instance(struct kprobe *kp)
{
 __this_cpu_write(kprobe_instance, kp);
}

static inline void reset_kprobe_instance(void)
{
 __this_cpu_write(kprobe_instance, NULL);
}







struct kprobe *get_kprobe(void *addr)
{
 struct hlist_head *head;
 struct kprobe *p;

 head = &kprobe_table[hash_ptr(addr, KPROBE_HASH_BITS)];
 hlist_for_each_entry_rcu(p, head, hlist) {
  if (p->addr == addr)
   return p;
 }

 return NULL;
}
NOKPROBE_SYMBOL(get_kprobe);

static int aggr_pre_handler(struct kprobe *p, struct pt_regs *regs);


static inline int kprobe_aggrprobe(struct kprobe *p)
{
 return p->pre_handler == aggr_pre_handler;
}


static inline int kprobe_unused(struct kprobe *p)
{
 return kprobe_aggrprobe(p) && kprobe_disabled(p) &&
        list_empty(&p->list);
}




static inline void copy_kprobe(struct kprobe *ap, struct kprobe *p)
{
 memcpy(&p->opcode, &ap->opcode, sizeof(kprobe_opcode_t));
 memcpy(&p->ainsn, &ap->ainsn, sizeof(struct arch_specific_insn));
}


static bool kprobes_allow_optimization;





void opt_pre_handler(struct kprobe *p, struct pt_regs *regs)
{
 struct kprobe *kp;

 list_for_each_entry_rcu(kp, &p->list, list) {
  if (kp->pre_handler && likely(!kprobe_disabled(kp))) {
   set_kprobe_instance(kp);
   kp->pre_handler(kp, regs);
  }
  reset_kprobe_instance();
 }
}
NOKPROBE_SYMBOL(opt_pre_handler);


static void free_aggr_kprobe(struct kprobe *p)
{
 struct optimized_kprobe *op;

 op = container_of(p, struct optimized_kprobe, kp);
 arch_remove_optimized_kprobe(op);
 arch_remove_kprobe(p);
 kfree(op);
}


static inline int kprobe_optready(struct kprobe *p)
{
 struct optimized_kprobe *op;

 if (kprobe_aggrprobe(p)) {
  op = container_of(p, struct optimized_kprobe, kp);
  return arch_prepared_optinsn(&op->optinsn);
 }

 return 0;
}


static inline int kprobe_disarmed(struct kprobe *p)
{
 struct optimized_kprobe *op;


 if (!kprobe_aggrprobe(p))
  return kprobe_disabled(p);

 op = container_of(p, struct optimized_kprobe, kp);

 return kprobe_disabled(p) && list_empty(&op->list);
}


static int kprobe_queued(struct kprobe *p)
{
 struct optimized_kprobe *op;

 if (kprobe_aggrprobe(p)) {
  op = container_of(p, struct optimized_kprobe, kp);
  if (!list_empty(&op->list))
   return 1;
 }
 return 0;
}





static struct kprobe *get_optimized_kprobe(unsigned long addr)
{
 int i;
 struct kprobe *p = NULL;
 struct optimized_kprobe *op;


 for (i = 1; !p && i < MAX_OPTIMIZED_LENGTH; i++)
  p = get_kprobe((void *)(addr - i));

 if (p && kprobe_optready(p)) {
  op = container_of(p, struct optimized_kprobe, kp);
  if (arch_within_optimized_kprobe(op, addr))
   return p;
 }

 return NULL;
}


static LIST_HEAD(optimizing_list);
static LIST_HEAD(unoptimizing_list);
static LIST_HEAD(freeing_list);

static void kprobe_optimizer(struct work_struct *work);
static DECLARE_DELAYED_WORK(optimizing_work, kprobe_optimizer);





static void do_optimize_kprobes(void)
{

 if (kprobes_all_disarmed || !kprobes_allow_optimization ||
     list_empty(&optimizing_list))
  return;
 get_online_cpus();
 mutex_lock(&text_mutex);
 arch_optimize_kprobes(&optimizing_list);
 mutex_unlock(&text_mutex);
 put_online_cpus();
}





static void do_unoptimize_kprobes(void)
{
 struct optimized_kprobe *op, *tmp;


 if (list_empty(&unoptimizing_list))
  return;


 get_online_cpus();
 mutex_lock(&text_mutex);
 arch_unoptimize_kprobes(&unoptimizing_list, &freeing_list);

 list_for_each_entry_safe(op, tmp, &freeing_list, list) {

  if (kprobe_disabled(&op->kp))
   arch_disarm_kprobe(&op->kp);
  if (kprobe_unused(&op->kp)) {





   hlist_del_rcu(&op->kp.hlist);
  } else
   list_del_init(&op->list);
 }
 mutex_unlock(&text_mutex);
 put_online_cpus();
}


static void do_free_cleaned_kprobes(void)
{
 struct optimized_kprobe *op, *tmp;

 list_for_each_entry_safe(op, tmp, &freeing_list, list) {
  BUG_ON(!kprobe_unused(&op->kp));
  list_del_init(&op->list);
  free_aggr_kprobe(&op->kp);
 }
}


static void kick_kprobe_optimizer(void)
{
 schedule_delayed_work(&optimizing_work, OPTIMIZE_DELAY);
}


static void kprobe_optimizer(struct work_struct *work)
{
 mutex_lock(&kprobe_mutex);

 mutex_lock(&module_mutex);





 do_unoptimize_kprobes();
 synchronize_sched();


 do_optimize_kprobes();


 do_free_cleaned_kprobes();

 mutex_unlock(&module_mutex);
 mutex_unlock(&kprobe_mutex);


 if (!list_empty(&optimizing_list) || !list_empty(&unoptimizing_list))
  kick_kprobe_optimizer();
}


static void wait_for_kprobe_optimizer(void)
{
 mutex_lock(&kprobe_mutex);

 while (!list_empty(&optimizing_list) || !list_empty(&unoptimizing_list)) {
  mutex_unlock(&kprobe_mutex);


  flush_delayed_work(&optimizing_work);

  cpu_relax();

  mutex_lock(&kprobe_mutex);
 }

 mutex_unlock(&kprobe_mutex);
}


static void optimize_kprobe(struct kprobe *p)
{
 struct optimized_kprobe *op;


 if (!kprobe_optready(p) || !kprobes_allow_optimization ||
     (kprobe_disabled(p) || kprobes_all_disarmed))
  return;


 if (p->break_handler || p->post_handler)
  return;

 op = container_of(p, struct optimized_kprobe, kp);


 if (arch_check_optimized_kprobe(op) < 0)
  return;


 if (op->kp.flags & KPROBE_FLAG_OPTIMIZED)
  return;
 op->kp.flags |= KPROBE_FLAG_OPTIMIZED;

 if (!list_empty(&op->list))

  list_del_init(&op->list);
 else {
  list_add(&op->list, &optimizing_list);
  kick_kprobe_optimizer();
 }
}


static void force_unoptimize_kprobe(struct optimized_kprobe *op)
{
 get_online_cpus();
 arch_unoptimize_kprobe(op);
 put_online_cpus();
 if (kprobe_disabled(&op->kp))
  arch_disarm_kprobe(&op->kp);
}


static void unoptimize_kprobe(struct kprobe *p, bool force)
{
 struct optimized_kprobe *op;

 if (!kprobe_aggrprobe(p) || kprobe_disarmed(p))
  return;

 op = container_of(p, struct optimized_kprobe, kp);
 if (!kprobe_optimized(p)) {

  if (force && !list_empty(&op->list)) {





   list_del_init(&op->list);
   force_unoptimize_kprobe(op);
  }
  return;
 }

 op->kp.flags &= ~KPROBE_FLAG_OPTIMIZED;
 if (!list_empty(&op->list)) {

  list_del_init(&op->list);
  return;
 }

 if (force)

  force_unoptimize_kprobe(op);
 else {
  list_add(&op->list, &unoptimizing_list);
  kick_kprobe_optimizer();
 }
}


static void reuse_unused_kprobe(struct kprobe *ap)
{
 struct optimized_kprobe *op;

 BUG_ON(!kprobe_unused(ap));




 op = container_of(ap, struct optimized_kprobe, kp);
 if (unlikely(list_empty(&op->list)))
  printk(KERN_WARNING "Warning: found a stray unused "
   "aggrprobe@%p\n", ap->addr);

 ap->flags &= ~KPROBE_FLAG_DISABLED;

 BUG_ON(!kprobe_optready(ap));
 optimize_kprobe(ap);
}


static void kill_optimized_kprobe(struct kprobe *p)
{
 struct optimized_kprobe *op;

 op = container_of(p, struct optimized_kprobe, kp);
 if (!list_empty(&op->list))

  list_del_init(&op->list);
 op->kp.flags &= ~KPROBE_FLAG_OPTIMIZED;

 if (kprobe_unused(p)) {

  list_add(&op->list, &freeing_list);





  hlist_del_rcu(&op->kp.hlist);
 }


 arch_remove_optimized_kprobe(op);
}


static void prepare_optimized_kprobe(struct kprobe *p)
{
 struct optimized_kprobe *op;

 op = container_of(p, struct optimized_kprobe, kp);
 arch_prepare_optimized_kprobe(op, p);
}


static struct kprobe *alloc_aggr_kprobe(struct kprobe *p)
{
 struct optimized_kprobe *op;

 op = kzalloc(sizeof(struct optimized_kprobe), GFP_KERNEL);
 if (!op)
  return NULL;

 INIT_LIST_HEAD(&op->list);
 op->kp.addr = p->addr;
 arch_prepare_optimized_kprobe(op, p);

 return &op->kp;
}

static void init_aggr_kprobe(struct kprobe *ap, struct kprobe *p);





static void try_to_optimize_kprobe(struct kprobe *p)
{
 struct kprobe *ap;
 struct optimized_kprobe *op;


 if (kprobe_ftrace(p))
  return;


 jump_label_lock();
 mutex_lock(&text_mutex);

 ap = alloc_aggr_kprobe(p);
 if (!ap)
  goto out;

 op = container_of(ap, struct optimized_kprobe, kp);
 if (!arch_prepared_optinsn(&op->optinsn)) {

  arch_remove_optimized_kprobe(op);
  kfree(op);
  goto out;
 }

 init_aggr_kprobe(ap, p);
 optimize_kprobe(ap);

out:
 mutex_unlock(&text_mutex);
 jump_label_unlock();
}

static void optimize_all_kprobes(void)
{
 struct hlist_head *head;
 struct kprobe *p;
 unsigned int i;

 mutex_lock(&kprobe_mutex);

 if (kprobes_allow_optimization)
  goto out;

 kprobes_allow_optimization = true;
 for (i = 0; i < KPROBE_TABLE_SIZE; i++) {
  head = &kprobe_table[i];
  hlist_for_each_entry_rcu(p, head, hlist)
   if (!kprobe_disabled(p))
    optimize_kprobe(p);
 }
 printk(KERN_INFO "Kprobes globally optimized\n");
out:
 mutex_unlock(&kprobe_mutex);
}

static void unoptimize_all_kprobes(void)
{
 struct hlist_head *head;
 struct kprobe *p;
 unsigned int i;

 mutex_lock(&kprobe_mutex);

 if (!kprobes_allow_optimization) {
  mutex_unlock(&kprobe_mutex);
  return;
 }

 kprobes_allow_optimization = false;
 for (i = 0; i < KPROBE_TABLE_SIZE; i++) {
  head = &kprobe_table[i];
  hlist_for_each_entry_rcu(p, head, hlist) {
   if (!kprobe_disabled(p))
    unoptimize_kprobe(p, false);
  }
 }
 mutex_unlock(&kprobe_mutex);


 wait_for_kprobe_optimizer();
 printk(KERN_INFO "Kprobes globally unoptimized\n");
}

static DEFINE_MUTEX(kprobe_sysctl_mutex);
int sysctl_kprobes_optimization;
int proc_kprobes_optimization_handler(struct ctl_table *table, int write,
          void __user *buffer, size_t *length,
          loff_t *ppos)
{
 int ret;

 mutex_lock(&kprobe_sysctl_mutex);
 sysctl_kprobes_optimization = kprobes_allow_optimization ? 1 : 0;
 ret = proc_dointvec_minmax(table, write, buffer, length, ppos);

 if (sysctl_kprobes_optimization)
  optimize_all_kprobes();
 else
  unoptimize_all_kprobes();
 mutex_unlock(&kprobe_sysctl_mutex);

 return ret;
}


static void __arm_kprobe(struct kprobe *p)
{
 struct kprobe *_p;


 _p = get_optimized_kprobe((unsigned long)p->addr);
 if (unlikely(_p))

  unoptimize_kprobe(_p, true);

 arch_arm_kprobe(p);
 optimize_kprobe(p);
}


static void __disarm_kprobe(struct kprobe *p, bool reopt)
{
 struct kprobe *_p;


 unoptimize_kprobe(p, kprobes_all_disarmed);

 if (!kprobe_queued(p)) {
  arch_disarm_kprobe(p);

  _p = get_optimized_kprobe((unsigned long)p->addr);
  if (unlikely(_p) && reopt)
   optimize_kprobe(_p);
 }

}




static void reuse_unused_kprobe(struct kprobe *ap)
{
 printk(KERN_ERR "Error: There should be no unused kprobe here.\n");
 BUG_ON(kprobe_unused(ap));
}

static void free_aggr_kprobe(struct kprobe *p)
{
 arch_remove_kprobe(p);
 kfree(p);
}

static struct kprobe *alloc_aggr_kprobe(struct kprobe *p)
{
 return kzalloc(sizeof(struct kprobe), GFP_KERNEL);
}

static struct ftrace_ops kprobe_ftrace_ops __read_mostly = {
 .func = kprobe_ftrace_handler,
 .flags = FTRACE_OPS_FL_SAVE_REGS | FTRACE_OPS_FL_IPMODIFY,
};
static int kprobe_ftrace_enabled;


static int prepare_kprobe(struct kprobe *p)
{
 if (!kprobe_ftrace(p))
  return arch_prepare_kprobe(p);

 return arch_prepare_kprobe_ftrace(p);
}


static void arm_kprobe_ftrace(struct kprobe *p)
{
 int ret;

 ret = ftrace_set_filter_ip(&kprobe_ftrace_ops,
       (unsigned long)p->addr, 0, 0);
 WARN(ret < 0, "Failed to arm kprobe-ftrace at %p (%d)\n", p->addr, ret);
 kprobe_ftrace_enabled++;
 if (kprobe_ftrace_enabled == 1) {
  ret = register_ftrace_function(&kprobe_ftrace_ops);
  WARN(ret < 0, "Failed to init kprobe-ftrace (%d)\n", ret);
 }
}


static void disarm_kprobe_ftrace(struct kprobe *p)
{
 int ret;

 kprobe_ftrace_enabled--;
 if (kprobe_ftrace_enabled == 0) {
  ret = unregister_ftrace_function(&kprobe_ftrace_ops);
  WARN(ret < 0, "Failed to init kprobe-ftrace (%d)\n", ret);
 }
 ret = ftrace_set_filter_ip(&kprobe_ftrace_ops,
      (unsigned long)p->addr, 1, 0);
 WARN(ret < 0, "Failed to disarm kprobe-ftrace at %p (%d)\n", p->addr, ret);
}


static void arm_kprobe(struct kprobe *kp)
{
 if (unlikely(kprobe_ftrace(kp))) {
  arm_kprobe_ftrace(kp);
  return;
 }





 mutex_lock(&text_mutex);
 __arm_kprobe(kp);
 mutex_unlock(&text_mutex);
}


static void disarm_kprobe(struct kprobe *kp, bool reopt)
{
 if (unlikely(kprobe_ftrace(kp))) {
  disarm_kprobe_ftrace(kp);
  return;
 }

 mutex_lock(&text_mutex);
 __disarm_kprobe(kp, reopt);
 mutex_unlock(&text_mutex);
}





static int aggr_pre_handler(struct kprobe *p, struct pt_regs *regs)
{
 struct kprobe *kp;

 list_for_each_entry_rcu(kp, &p->list, list) {
  if (kp->pre_handler && likely(!kprobe_disabled(kp))) {
   set_kprobe_instance(kp);
   if (kp->pre_handler(kp, regs))
    return 1;
  }
  reset_kprobe_instance();
 }
 return 0;
}
NOKPROBE_SYMBOL(aggr_pre_handler);

static void aggr_post_handler(struct kprobe *p, struct pt_regs *regs,
         unsigned long flags)
{
 struct kprobe *kp;

 list_for_each_entry_rcu(kp, &p->list, list) {
  if (kp->post_handler && likely(!kprobe_disabled(kp))) {
   set_kprobe_instance(kp);
   kp->post_handler(kp, regs, flags);
   reset_kprobe_instance();
  }
 }
}
NOKPROBE_SYMBOL(aggr_post_handler);

static int aggr_fault_handler(struct kprobe *p, struct pt_regs *regs,
         int trapnr)
{
 struct kprobe *cur = __this_cpu_read(kprobe_instance);





 if (cur && cur->fault_handler) {
  if (cur->fault_handler(cur, regs, trapnr))
   return 1;
 }
 return 0;
}
NOKPROBE_SYMBOL(aggr_fault_handler);

static int aggr_break_handler(struct kprobe *p, struct pt_regs *regs)
{
 struct kprobe *cur = __this_cpu_read(kprobe_instance);
 int ret = 0;

 if (cur && cur->break_handler) {
  if (cur->break_handler(cur, regs))
   ret = 1;
 }
 reset_kprobe_instance();
 return ret;
}
NOKPROBE_SYMBOL(aggr_break_handler);


void kprobes_inc_nmissed_count(struct kprobe *p)
{
 struct kprobe *kp;
 if (!kprobe_aggrprobe(p)) {
  p->nmissed++;
 } else {
  list_for_each_entry_rcu(kp, &p->list, list)
   kp->nmissed++;
 }
 return;
}
NOKPROBE_SYMBOL(kprobes_inc_nmissed_count);

void recycle_rp_inst(struct kretprobe_instance *ri,
       struct hlist_head *head)
{
 struct kretprobe *rp = ri->rp;


 hlist_del(&ri->hlist);
 INIT_HLIST_NODE(&ri->hlist);
 if (likely(rp)) {
  raw_spin_lock(&rp->lock);
  hlist_add_head(&ri->hlist, &rp->free_instances);
  raw_spin_unlock(&rp->lock);
 } else

  hlist_add_head(&ri->hlist, head);
}
NOKPROBE_SYMBOL(recycle_rp_inst);

void kretprobe_hash_lock(struct task_struct *tsk,
    struct hlist_head **head, unsigned long *flags)
__acquires(hlist_lock)
{
 unsigned long hash = hash_ptr(tsk, KPROBE_HASH_BITS);
 raw_spinlock_t *hlist_lock;

 *head = &kretprobe_inst_table[hash];
 hlist_lock = kretprobe_table_lock_ptr(hash);
 raw_spin_lock_irqsave(hlist_lock, *flags);
}
NOKPROBE_SYMBOL(kretprobe_hash_lock);

static void kretprobe_table_lock(unsigned long hash,
     unsigned long *flags)
__acquires(hlist_lock)
{
 raw_spinlock_t *hlist_lock = kretprobe_table_lock_ptr(hash);
 raw_spin_lock_irqsave(hlist_lock, *flags);
}
NOKPROBE_SYMBOL(kretprobe_table_lock);

void kretprobe_hash_unlock(struct task_struct *tsk,
      unsigned long *flags)
__releases(hlist_lock)
{
 unsigned long hash = hash_ptr(tsk, KPROBE_HASH_BITS);
 raw_spinlock_t *hlist_lock;

 hlist_lock = kretprobe_table_lock_ptr(hash);
 raw_spin_unlock_irqrestore(hlist_lock, *flags);
}
NOKPROBE_SYMBOL(kretprobe_hash_unlock);

static void kretprobe_table_unlock(unsigned long hash,
       unsigned long *flags)
__releases(hlist_lock)
{
 raw_spinlock_t *hlist_lock = kretprobe_table_lock_ptr(hash);
 raw_spin_unlock_irqrestore(hlist_lock, *flags);
}
NOKPROBE_SYMBOL(kretprobe_table_unlock);







void kprobe_flush_task(struct task_struct *tk)
{
 struct kretprobe_instance *ri;
 struct hlist_head *head, empty_rp;
 struct hlist_node *tmp;
 unsigned long hash, flags = 0;

 if (unlikely(!kprobes_initialized))

  return;

 INIT_HLIST_HEAD(&empty_rp);
 hash = hash_ptr(tk, KPROBE_HASH_BITS);
 head = &kretprobe_inst_table[hash];
 kretprobe_table_lock(hash, &flags);
 hlist_for_each_entry_safe(ri, tmp, head, hlist) {
  if (ri->task == tk)
   recycle_rp_inst(ri, &empty_rp);
 }
 kretprobe_table_unlock(hash, &flags);
 hlist_for_each_entry_safe(ri, tmp, &empty_rp, hlist) {
  hlist_del(&ri->hlist);
  kfree(ri);
 }
}
NOKPROBE_SYMBOL(kprobe_flush_task);

static inline void free_rp_inst(struct kretprobe *rp)
{
 struct kretprobe_instance *ri;
 struct hlist_node *next;

 hlist_for_each_entry_safe(ri, next, &rp->free_instances, hlist) {
  hlist_del(&ri->hlist);
  kfree(ri);
 }
}

static void cleanup_rp_inst(struct kretprobe *rp)
{
 unsigned long flags, hash;
 struct kretprobe_instance *ri;
 struct hlist_node *next;
 struct hlist_head *head;


 for (hash = 0; hash < KPROBE_TABLE_SIZE; hash++) {
  kretprobe_table_lock(hash, &flags);
  head = &kretprobe_inst_table[hash];
  hlist_for_each_entry_safe(ri, next, head, hlist) {
   if (ri->rp == rp)
    ri->rp = NULL;
  }
  kretprobe_table_unlock(hash, &flags);
 }
 free_rp_inst(rp);
}
NOKPROBE_SYMBOL(cleanup_rp_inst);





static int add_new_kprobe(struct kprobe *ap, struct kprobe *p)
{
 BUG_ON(kprobe_gone(ap) || kprobe_gone(p));

 if (p->break_handler || p->post_handler)
  unoptimize_kprobe(ap, true);

 if (p->break_handler) {
  if (ap->break_handler)
   return -EEXIST;
  list_add_tail_rcu(&p->list, &ap->list);
  ap->break_handler = aggr_break_handler;
 } else
  list_add_rcu(&p->list, &ap->list);
 if (p->post_handler && !ap->post_handler)
  ap->post_handler = aggr_post_handler;

 return 0;
}





static void init_aggr_kprobe(struct kprobe *ap, struct kprobe *p)
{

 copy_kprobe(p, ap);
 flush_insn_slot(ap);
 ap->addr = p->addr;
 ap->flags = p->flags & ~KPROBE_FLAG_OPTIMIZED;
 ap->pre_handler = aggr_pre_handler;
 ap->fault_handler = aggr_fault_handler;

 if (p->post_handler && !kprobe_gone(p))
  ap->post_handler = aggr_post_handler;
 if (p->break_handler && !kprobe_gone(p))
  ap->break_handler = aggr_break_handler;

 INIT_LIST_HEAD(&ap->list);
 INIT_HLIST_NODE(&ap->hlist);

 list_add_rcu(&p->list, &ap->list);
 hlist_replace_rcu(&p->hlist, &ap->hlist);
}





static int register_aggr_kprobe(struct kprobe *orig_p, struct kprobe *p)
{
 int ret = 0;
 struct kprobe *ap = orig_p;


 jump_label_lock();




 get_online_cpus();
 mutex_lock(&text_mutex);

 if (!kprobe_aggrprobe(orig_p)) {

  ap = alloc_aggr_kprobe(orig_p);
  if (!ap) {
   ret = -ENOMEM;
   goto out;
  }
  init_aggr_kprobe(ap, orig_p);
 } else if (kprobe_unused(ap))

  reuse_unused_kprobe(ap);

 if (kprobe_gone(ap)) {






  ret = arch_prepare_kprobe(ap);
  if (ret)





   goto out;


  prepare_optimized_kprobe(ap);





  ap->flags = (ap->flags & ~KPROBE_FLAG_GONE)
       | KPROBE_FLAG_DISABLED;
 }


 copy_kprobe(ap, p);
 ret = add_new_kprobe(ap, p);

out:
 mutex_unlock(&text_mutex);
 put_online_cpus();
 jump_label_unlock();

 if (ret == 0 && kprobe_disabled(ap) && !kprobe_disabled(p)) {
  ap->flags &= ~KPROBE_FLAG_DISABLED;
  if (!kprobes_all_disarmed)

   arm_kprobe(ap);
 }
 return ret;
}

bool __weak arch_within_kprobe_blacklist(unsigned long addr)
{

 return addr >= (unsigned long)__kprobes_text_start &&
        addr < (unsigned long)__kprobes_text_end;
}

bool within_kprobe_blacklist(unsigned long addr)
{
 struct kprobe_blacklist_entry *ent;

 if (arch_within_kprobe_blacklist(addr))
  return true;




 list_for_each_entry(ent, &kprobe_blacklist, list) {
  if (addr >= ent->start_addr && addr < ent->end_addr)
   return true;
 }

 return false;
}







static kprobe_opcode_t *kprobe_addr(struct kprobe *p)
{
 kprobe_opcode_t *addr = p->addr;

 if ((p->symbol_name && p->addr) ||
     (!p->symbol_name && !p->addr))
  goto invalid;

 if (p->symbol_name) {
  kprobe_lookup_name(p->symbol_name, addr);
  if (!addr)
   return ERR_PTR(-ENOENT);
 }

 addr = (kprobe_opcode_t *)(((char *)addr) + p->offset);
 if (addr)
  return addr;

invalid:
 return ERR_PTR(-EINVAL);
}


static struct kprobe *__get_valid_kprobe(struct kprobe *p)
{
 struct kprobe *ap, *list_p;

 ap = get_kprobe(p->addr);
 if (unlikely(!ap))
  return NULL;

 if (p != ap) {
  list_for_each_entry_rcu(list_p, &ap->list, list)
   if (list_p == p)

    goto valid;
  return NULL;
 }
valid:
 return ap;
}


static inline int check_kprobe_rereg(struct kprobe *p)
{
 int ret = 0;

 mutex_lock(&kprobe_mutex);
 if (__get_valid_kprobe(p))
  ret = -EINVAL;
 mutex_unlock(&kprobe_mutex);

 return ret;
}

int __weak arch_check_ftrace_location(struct kprobe *p)
{
 unsigned long ftrace_addr;

 ftrace_addr = ftrace_location((unsigned long)p->addr);
 if (ftrace_addr) {

  if ((unsigned long)p->addr != ftrace_addr)
   return -EILSEQ;
  p->flags |= KPROBE_FLAG_FTRACE;
  return -EINVAL;
 }
 return 0;
}

static int check_kprobe_address_safe(struct kprobe *p,
         struct module **probed_mod)
{
 int ret;

 ret = arch_check_ftrace_location(p);
 if (ret)
  return ret;
 jump_label_lock();
 preempt_disable();


 if (!kernel_text_address((unsigned long) p->addr) ||
     within_kprobe_blacklist((unsigned long) p->addr) ||
     jump_label_text_reserved(p->addr, p->addr)) {
  ret = -EINVAL;
  goto out;
 }


 *probed_mod = __module_text_address((unsigned long) p->addr);
 if (*probed_mod) {




  if (unlikely(!try_module_get(*probed_mod))) {
   ret = -ENOENT;
   goto out;
  }





  if (within_module_init((unsigned long)p->addr, *probed_mod) &&
      (*probed_mod)->state != MODULE_STATE_COMING) {
   module_put(*probed_mod);
   *probed_mod = NULL;
   ret = -ENOENT;
  }
 }
out:
 preempt_enable();
 jump_label_unlock();

 return ret;
}

int register_kprobe(struct kprobe *p)
{
 int ret;
 struct kprobe *old_p;
 struct module *probed_mod;
 kprobe_opcode_t *addr;


 addr = kprobe_addr(p);
 if (IS_ERR(addr))
  return PTR_ERR(addr);
 p->addr = addr;

 ret = check_kprobe_rereg(p);
 if (ret)
  return ret;


 p->flags &= KPROBE_FLAG_DISABLED;
 p->nmissed = 0;
 INIT_LIST_HEAD(&p->list);

 ret = check_kprobe_address_safe(p, &probed_mod);
 if (ret)
  return ret;

 mutex_lock(&kprobe_mutex);

 old_p = get_kprobe(p->addr);
 if (old_p) {

  ret = register_aggr_kprobe(old_p, p);
  goto out;
 }

 mutex_lock(&text_mutex);
 ret = prepare_kprobe(p);
 mutex_unlock(&text_mutex);
 if (ret)
  goto out;

 INIT_HLIST_NODE(&p->hlist);
 hlist_add_head_rcu(&p->hlist,
         &kprobe_table[hash_ptr(p->addr, KPROBE_HASH_BITS)]);

 if (!kprobes_all_disarmed && !kprobe_disabled(p))
  arm_kprobe(p);


 try_to_optimize_kprobe(p);

out:
 mutex_unlock(&kprobe_mutex);

 if (probed_mod)
  module_put(probed_mod);

 return ret;
}
EXPORT_SYMBOL_GPL(register_kprobe);


static int aggr_kprobe_disabled(struct kprobe *ap)
{
 struct kprobe *kp;

 list_for_each_entry_rcu(kp, &ap->list, list)
  if (!kprobe_disabled(kp))




   return 0;

 return 1;
}


static struct kprobe *__disable_kprobe(struct kprobe *p)
{
 struct kprobe *orig_p;


 orig_p = __get_valid_kprobe(p);
 if (unlikely(orig_p == NULL))
  return NULL;

 if (!kprobe_disabled(p)) {

  if (p != orig_p)
   p->flags |= KPROBE_FLAG_DISABLED;


  if (p == orig_p || aggr_kprobe_disabled(orig_p)) {





   if (!kprobes_all_disarmed)
    disarm_kprobe(orig_p, true);
   orig_p->flags |= KPROBE_FLAG_DISABLED;
  }
 }

 return orig_p;
}




static int __unregister_kprobe_top(struct kprobe *p)
{
 struct kprobe *ap, *list_p;


 ap = __disable_kprobe(p);
 if (ap == NULL)
  return -EINVAL;

 if (ap == p)




  goto disarmed;


 WARN_ON(!kprobe_aggrprobe(ap));

 if (list_is_singular(&ap->list) && kprobe_disarmed(ap))




  goto disarmed;
 else {

  if (p->break_handler && !kprobe_gone(p))
   ap->break_handler = NULL;
  if (p->post_handler && !kprobe_gone(p)) {
   list_for_each_entry_rcu(list_p, &ap->list, list) {
    if ((list_p != p) && (list_p->post_handler))
     goto noclean;
   }
   ap->post_handler = NULL;
  }
noclean:




  list_del_rcu(&p->list);
  if (!kprobe_disabled(ap) && !kprobes_all_disarmed)




   optimize_kprobe(ap);
 }
 return 0;

disarmed:
 BUG_ON(!kprobe_disarmed(ap));
 hlist_del_rcu(&ap->hlist);
 return 0;
}

static void __unregister_kprobe_bottom(struct kprobe *p)
{
 struct kprobe *ap;

 if (list_empty(&p->list))

  arch_remove_kprobe(p);
 else if (list_is_singular(&p->list)) {

  ap = list_entry(p->list.next, struct kprobe, list);
  list_del(&p->list);
  free_aggr_kprobe(ap);
 }

}

int register_kprobes(struct kprobe **kps, int num)
{
 int i, ret = 0;

 if (num <= 0)
  return -EINVAL;
 for (i = 0; i < num; i++) {
  ret = register_kprobe(kps[i]);
  if (ret < 0) {
   if (i > 0)
    unregister_kprobes(kps, i);
   break;
  }
 }
 return ret;
}
EXPORT_SYMBOL_GPL(register_kprobes);

void unregister_kprobe(struct kprobe *p)
{
 unregister_kprobes(&p, 1);
}
EXPORT_SYMBOL_GPL(unregister_kprobe);

void unregister_kprobes(struct kprobe **kps, int num)
{
 int i;

 if (num <= 0)
  return;
 mutex_lock(&kprobe_mutex);
 for (i = 0; i < num; i++)
  if (__unregister_kprobe_top(kps[i]) < 0)
   kps[i]->addr = NULL;
 mutex_unlock(&kprobe_mutex);

 synchronize_sched();
 for (i = 0; i < num; i++)
  if (kps[i]->addr)
   __unregister_kprobe_bottom(kps[i]);
}
EXPORT_SYMBOL_GPL(unregister_kprobes);

static struct notifier_block kprobe_exceptions_nb = {
 .notifier_call = kprobe_exceptions_notify,
 .priority = 0x7fffffff
};

unsigned long __weak arch_deref_entry_point(void *entry)
{
 return (unsigned long)entry;
}

int register_jprobes(struct jprobe **jps, int num)
{
 struct jprobe *jp;
 int ret = 0, i;

 if (num <= 0)
  return -EINVAL;
 for (i = 0; i < num; i++) {
  unsigned long addr, offset;
  jp = jps[i];
  addr = arch_deref_entry_point(jp->entry);


  if (kallsyms_lookup_size_offset(addr, NULL, &offset) &&
      offset == 0) {
   jp->kp.pre_handler = setjmp_pre_handler;
   jp->kp.break_handler = longjmp_break_handler;
   ret = register_kprobe(&jp->kp);
  } else
   ret = -EINVAL;

  if (ret < 0) {
   if (i > 0)
    unregister_jprobes(jps, i);
   break;
  }
 }
 return ret;
}
EXPORT_SYMBOL_GPL(register_jprobes);

int register_jprobe(struct jprobe *jp)
{
 return register_jprobes(&jp, 1);
}
EXPORT_SYMBOL_GPL(register_jprobe);

void unregister_jprobe(struct jprobe *jp)
{
 unregister_jprobes(&jp, 1);
}
EXPORT_SYMBOL_GPL(unregister_jprobe);

void unregister_jprobes(struct jprobe **jps, int num)
{
 int i;

 if (num <= 0)
  return;
 mutex_lock(&kprobe_mutex);
 for (i = 0; i < num; i++)
  if (__unregister_kprobe_top(&jps[i]->kp) < 0)
   jps[i]->kp.addr = NULL;
 mutex_unlock(&kprobe_mutex);

 synchronize_sched();
 for (i = 0; i < num; i++) {
  if (jps[i]->kp.addr)
   __unregister_kprobe_bottom(&jps[i]->kp);
 }
}
EXPORT_SYMBOL_GPL(unregister_jprobes);





static int pre_handler_kretprobe(struct kprobe *p, struct pt_regs *regs)
{
 struct kretprobe *rp = container_of(p, struct kretprobe, kp);
 unsigned long hash, flags = 0;
 struct kretprobe_instance *ri;







 if (unlikely(in_nmi())) {
  rp->nmissed++;
  return 0;
 }


 hash = hash_ptr(current, KPROBE_HASH_BITS);
 raw_spin_lock_irqsave(&rp->lock, flags);
 if (!hlist_empty(&rp->free_instances)) {
  ri = hlist_entry(rp->free_instances.first,
    struct kretprobe_instance, hlist);
  hlist_del(&ri->hlist);
  raw_spin_unlock_irqrestore(&rp->lock, flags);

  ri->rp = rp;
  ri->task = current;

  if (rp->entry_handler && rp->entry_handler(ri, regs)) {
   raw_spin_lock_irqsave(&rp->lock, flags);
   hlist_add_head(&ri->hlist, &rp->free_instances);
   raw_spin_unlock_irqrestore(&rp->lock, flags);
   return 0;
  }

  arch_prepare_kretprobe(ri, regs);


  INIT_HLIST_NODE(&ri->hlist);
  kretprobe_table_lock(hash, &flags);
  hlist_add_head(&ri->hlist, &kretprobe_inst_table[hash]);
  kretprobe_table_unlock(hash, &flags);
 } else {
  rp->nmissed++;
  raw_spin_unlock_irqrestore(&rp->lock, flags);
 }
 return 0;
}
NOKPROBE_SYMBOL(pre_handler_kretprobe);

int register_kretprobe(struct kretprobe *rp)
{
 int ret = 0;
 struct kretprobe_instance *inst;
 int i;
 void *addr;

 if (kretprobe_blacklist_size) {
  addr = kprobe_addr(&rp->kp);
  if (IS_ERR(addr))
   return PTR_ERR(addr);

  for (i = 0; kretprobe_blacklist[i].name != NULL; i++) {
   if (kretprobe_blacklist[i].addr == addr)
    return -EINVAL;
  }
 }

 rp->kp.pre_handler = pre_handler_kretprobe;
 rp->kp.post_handler = NULL;
 rp->kp.fault_handler = NULL;
 rp->kp.break_handler = NULL;


 if (rp->maxactive <= 0) {
  rp->maxactive = max_t(unsigned int, 10, 2*num_possible_cpus());
  rp->maxactive = num_possible_cpus();
 }
 raw_spin_lock_init(&rp->lock);
 INIT_HLIST_HEAD(&rp->free_instances);
 for (i = 0; i < rp->maxactive; i++) {
  inst = kmalloc(sizeof(struct kretprobe_instance) +
          rp->data_size, GFP_KERNEL);
  if (inst == NULL) {
   free_rp_inst(rp);
   return -ENOMEM;
  }
  INIT_HLIST_NODE(&inst->hlist);
  hlist_add_head(&inst->hlist, &rp->free_instances);
 }

 rp->nmissed = 0;

 ret = register_kprobe(&rp->kp);
 if (ret != 0)
  free_rp_inst(rp);
 return ret;
}
EXPORT_SYMBOL_GPL(register_kretprobe);

int register_kretprobes(struct kretprobe **rps, int num)
{
 int ret = 0, i;

 if (num <= 0)
  return -EINVAL;
 for (i = 0; i < num; i++) {
  ret = register_kretprobe(rps[i]);
  if (ret < 0) {
   if (i > 0)
    unregister_kretprobes(rps, i);
   break;
  }
 }
 return ret;
}
EXPORT_SYMBOL_GPL(register_kretprobes);

void unregister_kretprobe(struct kretprobe *rp)
{
 unregister_kretprobes(&rp, 1);
}
EXPORT_SYMBOL_GPL(unregister_kretprobe);

void unregister_kretprobes(struct kretprobe **rps, int num)
{
 int i;

 if (num <= 0)
  return;
 mutex_lock(&kprobe_mutex);
 for (i = 0; i < num; i++)
  if (__unregister_kprobe_top(&rps[i]->kp) < 0)
   rps[i]->kp.addr = NULL;
 mutex_unlock(&kprobe_mutex);

 synchronize_sched();
 for (i = 0; i < num; i++) {
  if (rps[i]->kp.addr) {
   __unregister_kprobe_bottom(&rps[i]->kp);
   cleanup_rp_inst(rps[i]);
  }
 }
}
EXPORT_SYMBOL_GPL(unregister_kretprobes);

int register_kretprobe(struct kretprobe *rp)
{
 return -ENOSYS;
}
EXPORT_SYMBOL_GPL(register_kretprobe);

int register_kretprobes(struct kretprobe **rps, int num)
{
 return -ENOSYS;
}
EXPORT_SYMBOL_GPL(register_kretprobes);

void unregister_kretprobe(struct kretprobe *rp)
{
}
EXPORT_SYMBOL_GPL(unregister_kretprobe);

void unregister_kretprobes(struct kretprobe **rps, int num)
{
}
EXPORT_SYMBOL_GPL(unregister_kretprobes);

static int pre_handler_kretprobe(struct kprobe *p, struct pt_regs *regs)
{
 return 0;
}
NOKPROBE_SYMBOL(pre_handler_kretprobe);



static void kill_kprobe(struct kprobe *p)
{
 struct kprobe *kp;

 p->flags |= KPROBE_FLAG_GONE;
 if (kprobe_aggrprobe(p)) {




  list_for_each_entry_rcu(kp, &p->list, list)
   kp->flags |= KPROBE_FLAG_GONE;
  p->post_handler = NULL;
  p->break_handler = NULL;
  kill_optimized_kprobe(p);
 }




 arch_remove_kprobe(p);
}


int disable_kprobe(struct kprobe *kp)
{
 int ret = 0;

 mutex_lock(&kprobe_mutex);


 if (__disable_kprobe(kp) == NULL)
  ret = -EINVAL;

 mutex_unlock(&kprobe_mutex);
 return ret;
}
EXPORT_SYMBOL_GPL(disable_kprobe);


int enable_kprobe(struct kprobe *kp)
{
 int ret = 0;
 struct kprobe *p;

 mutex_lock(&kprobe_mutex);


 p = __get_valid_kprobe(kp);
 if (unlikely(p == NULL)) {
  ret = -EINVAL;
  goto out;
 }

 if (kprobe_gone(kp)) {

  ret = -EINVAL;
  goto out;
 }

 if (p != kp)
  kp->flags &= ~KPROBE_FLAG_DISABLED;

 if (!kprobes_all_disarmed && kprobe_disabled(p)) {
  p->flags &= ~KPROBE_FLAG_DISABLED;
  arm_kprobe(p);
 }
out:
 mutex_unlock(&kprobe_mutex);
 return ret;
}
EXPORT_SYMBOL_GPL(enable_kprobe);

void dump_kprobe(struct kprobe *kp)
{
 printk(KERN_WARNING "Dumping kprobe:\n");
 printk(KERN_WARNING "Name: %s\nAddress: %p\nOffset: %x\n",
        kp->symbol_name, kp->addr, kp->offset);
}
NOKPROBE_SYMBOL(dump_kprobe);
static int __init populate_kprobe_blacklist(unsigned long *start,
          unsigned long *end)
{
 unsigned long *iter;
 struct kprobe_blacklist_entry *ent;
 unsigned long entry, offset = 0, size = 0;

 for (iter = start; iter < end; iter++) {
  entry = arch_deref_entry_point((void *)*iter);

  if (!kernel_text_address(entry) ||
      !kallsyms_lookup_size_offset(entry, &size, &offset)) {
   pr_err("Failed to find blacklist at %p\n",
    (void *)entry);
   continue;
  }

  ent = kmalloc(sizeof(*ent), GFP_KERNEL);
  if (!ent)
   return -ENOMEM;
  ent->start_addr = entry;
  ent->end_addr = entry + size;
  INIT_LIST_HEAD(&ent->list);
  list_add_tail(&ent->list, &kprobe_blacklist);
 }
 return 0;
}


static int kprobes_module_callback(struct notifier_block *nb,
       unsigned long val, void *data)
{
 struct module *mod = data;
 struct hlist_head *head;
 struct kprobe *p;
 unsigned int i;
 int checkcore = (val == MODULE_STATE_GOING);

 if (val != MODULE_STATE_GOING && val != MODULE_STATE_LIVE)
  return NOTIFY_DONE;







 mutex_lock(&kprobe_mutex);
 for (i = 0; i < KPROBE_TABLE_SIZE; i++) {
  head = &kprobe_table[i];
  hlist_for_each_entry_rcu(p, head, hlist)
   if (within_module_init((unsigned long)p->addr, mod) ||
       (checkcore &&
        within_module_core((unsigned long)p->addr, mod))) {





    kill_kprobe(p);
   }
 }
 mutex_unlock(&kprobe_mutex);
 return NOTIFY_DONE;
}

static struct notifier_block kprobe_module_nb = {
 .notifier_call = kprobes_module_callback,
 .priority = 0
};


extern unsigned long __start_kprobe_blacklist[];
extern unsigned long __stop_kprobe_blacklist[];

static int __init init_kprobes(void)
{
 int i, err = 0;



 for (i = 0; i < KPROBE_TABLE_SIZE; i++) {
  INIT_HLIST_HEAD(&kprobe_table[i]);
  INIT_HLIST_HEAD(&kretprobe_inst_table[i]);
  raw_spin_lock_init(&(kretprobe_table_locks[i].lock));
 }

 err = populate_kprobe_blacklist(__start_kprobe_blacklist,
     __stop_kprobe_blacklist);
 if (err) {
  pr_err("kprobes: failed to populate blacklist: %d\n", err);
  pr_err("Please take care of using kprobes.\n");
 }

 if (kretprobe_blacklist_size) {

  for (i = 0; kretprobe_blacklist[i].name != NULL; i++) {
   kprobe_lookup_name(kretprobe_blacklist[i].name,
        kretprobe_blacklist[i].addr);
   if (!kretprobe_blacklist[i].addr)
    printk("kretprobe: lookup failed: %s\n",
           kretprobe_blacklist[i].name);
  }
 }


 kprobe_optinsn_slots.insn_size = MAX_OPTINSN_SIZE;

 kprobes_allow_optimization = true;


 kprobes_all_disarmed = false;

 err = arch_init_kprobes();
 if (!err)
  err = register_die_notifier(&kprobe_exceptions_nb);
 if (!err)
  err = register_module_notifier(&kprobe_module_nb);

 kprobes_initialized = (err == 0);

 if (!err)
  init_test_probes();
 return err;
}

static void report_probe(struct seq_file *pi, struct kprobe *p,
  const char *sym, int offset, char *modname, struct kprobe *pp)
{
 char *kprobe_type;

 if (p->pre_handler == pre_handler_kretprobe)
  kprobe_type = "r";
 else if (p->pre_handler == setjmp_pre_handler)
  kprobe_type = "j";
 else
  kprobe_type = "k";

 if (sym)
  seq_printf(pi, "%p  %s  %s+0x%x  %s ",
   p->addr, kprobe_type, sym, offset,
   (modname ? modname : " "));
 else
  seq_printf(pi, "%p  %s  %p ",
   p->addr, kprobe_type, p->addr);

 if (!pp)
  pp = p;
 seq_printf(pi, "%s%s%s%s\n",
  (kprobe_gone(p) ? "[GONE]" : ""),
  ((kprobe_disabled(p) && !kprobe_gone(p)) ? "[DISABLED]" : ""),
  (kprobe_optimized(pp) ? "[OPTIMIZED]" : ""),
  (kprobe_ftrace(pp) ? "[FTRACE]" : ""));
}

static void *kprobe_seq_start(struct seq_file *f, loff_t *pos)
{
 return (*pos < KPROBE_TABLE_SIZE) ? pos : NULL;
}

static void *kprobe_seq_next(struct seq_file *f, void *v, loff_t *pos)
{
 (*pos)++;
 if (*pos >= KPROBE_TABLE_SIZE)
  return NULL;
 return pos;
}

static void kprobe_seq_stop(struct seq_file *f, void *v)
{

}

static int show_kprobe_addr(struct seq_file *pi, void *v)
{
 struct hlist_head *head;
 struct kprobe *p, *kp;
 const char *sym = NULL;
 unsigned int i = *(loff_t *) v;
 unsigned long offset = 0;
 char *modname, namebuf[KSYM_NAME_LEN];

 head = &kprobe_table[i];
 preempt_disable();
 hlist_for_each_entry_rcu(p, head, hlist) {
  sym = kallsyms_lookup((unsigned long)p->addr, NULL,
     &offset, &modname, namebuf);
  if (kprobe_aggrprobe(p)) {
   list_for_each_entry_rcu(kp, &p->list, list)
    report_probe(pi, kp, sym, offset, modname, p);
  } else
   report_probe(pi, p, sym, offset, modname, NULL);
 }
 preempt_enable();
 return 0;
}

static const struct seq_operations kprobes_seq_ops = {
 .start = kprobe_seq_start,
 .next = kprobe_seq_next,
 .stop = kprobe_seq_stop,
 .show = show_kprobe_addr
};

static int kprobes_open(struct inode *inode, struct file *filp)
{
 return seq_open(filp, &kprobes_seq_ops);
}

static const struct file_operations debugfs_kprobes_operations = {
 .open = kprobes_open,
 .read = seq_read,
 .llseek = seq_lseek,
 .release = seq_release,
};


static void *kprobe_blacklist_seq_start(struct seq_file *m, loff_t *pos)
{
 return seq_list_start(&kprobe_blacklist, *pos);
}

static void *kprobe_blacklist_seq_next(struct seq_file *m, void *v, loff_t *pos)
{
 return seq_list_next(v, &kprobe_blacklist, pos);
}

static int kprobe_blacklist_seq_show(struct seq_file *m, void *v)
{
 struct kprobe_blacklist_entry *ent =
  list_entry(v, struct kprobe_blacklist_entry, list);

 seq_printf(m, "0x%p-0x%p\t%ps\n", (void *)ent->start_addr,
     (void *)ent->end_addr, (void *)ent->start_addr);
 return 0;
}

static const struct seq_operations kprobe_blacklist_seq_ops = {
 .start = kprobe_blacklist_seq_start,
 .next = kprobe_blacklist_seq_next,
 .stop = kprobe_seq_stop,
 .show = kprobe_blacklist_seq_show,
};

static int kprobe_blacklist_open(struct inode *inode, struct file *filp)
{
 return seq_open(filp, &kprobe_blacklist_seq_ops);
}

static const struct file_operations debugfs_kprobe_blacklist_ops = {
 .open = kprobe_blacklist_open,
 .read = seq_read,
 .llseek = seq_lseek,
 .release = seq_release,
};

static void arm_all_kprobes(void)
{
 struct hlist_head *head;
 struct kprobe *p;
 unsigned int i;

 mutex_lock(&kprobe_mutex);


 if (!kprobes_all_disarmed)
  goto already_enabled;






 kprobes_all_disarmed = false;

 for (i = 0; i < KPROBE_TABLE_SIZE; i++) {
  head = &kprobe_table[i];
  hlist_for_each_entry_rcu(p, head, hlist)
   if (!kprobe_disabled(p))
    arm_kprobe(p);
 }

 printk(KERN_INFO "Kprobes globally enabled\n");

already_enabled:
 mutex_unlock(&kprobe_mutex);
 return;
}

static void disarm_all_kprobes(void)
{
 struct hlist_head *head;
 struct kprobe *p;
 unsigned int i;

 mutex_lock(&kprobe_mutex);


 if (kprobes_all_disarmed) {
  mutex_unlock(&kprobe_mutex);
  return;
 }

 kprobes_all_disarmed = true;
 printk(KERN_INFO "Kprobes globally disabled\n");

 for (i = 0; i < KPROBE_TABLE_SIZE; i++) {
  head = &kprobe_table[i];
  hlist_for_each_entry_rcu(p, head, hlist) {
   if (!arch_trampoline_kprobe(p) && !kprobe_disabled(p))
    disarm_kprobe(p, false);
  }
 }
 mutex_unlock(&kprobe_mutex);


 wait_for_kprobe_optimizer();
}






static ssize_t read_enabled_file_bool(struct file *file,
        char __user *user_buf, size_t count, loff_t *ppos)
{
 char buf[3];

 if (!kprobes_all_disarmed)
  buf[0] = '1';
 else
  buf[0] = '0';
 buf[1] = '\n';
 buf[2] = 0x00;
 return simple_read_from_buffer(user_buf, count, ppos, buf, 2);
}

static ssize_t write_enabled_file_bool(struct file *file,
        const char __user *user_buf, size_t count, loff_t *ppos)
{
 char buf[32];
 size_t buf_size;

 buf_size = min(count, (sizeof(buf)-1));
 if (copy_from_user(buf, user_buf, buf_size))
  return -EFAULT;

 buf[buf_size] = '\0';
 switch (buf[0]) {
 case 'y':
 case 'Y':
 case '1':
  arm_all_kprobes();
  break;
 case 'n':
 case 'N':
 case '0':
  disarm_all_kprobes();
  break;
 default:
  return -EINVAL;
 }

 return count;
}

static const struct file_operations fops_kp = {
 .read = read_enabled_file_bool,
 .write = write_enabled_file_bool,
 .llseek = default_llseek,
};

static int __init debugfs_kprobe_init(void)
{
 struct dentry *dir, *file;
 unsigned int value = 1;

 dir = debugfs_create_dir("kprobes", NULL);
 if (!dir)
  return -ENOMEM;

 file = debugfs_create_file("list", 0444, dir, NULL,
    &debugfs_kprobes_operations);
 if (!file)
  goto error;

 file = debugfs_create_file("enabled", 0600, dir,
     &value, &fops_kp);
 if (!file)
  goto error;

 file = debugfs_create_file("blacklist", 0444, dir, NULL,
    &debugfs_kprobe_blacklist_ops);
 if (!file)
  goto error;

 return 0;

error:
 debugfs_remove(dir);
 return -ENOMEM;
}

late_initcall(debugfs_kprobe_init);

module_init(init_kprobes);


EXPORT_SYMBOL_GPL(jprobe_return);


static struct kobj_attribute _name##_attr = __ATTR_RO(_name)

static struct kobj_attribute _name##_attr = \
 __ATTR(_name, 0644, _name##_show, _name##_store)


static ssize_t uevent_seqnum_show(struct kobject *kobj,
      struct kobj_attribute *attr, char *buf)
{
 return sprintf(buf, "%llu\n", (unsigned long long)uevent_seqnum);
}
KERNEL_ATTR_RO(uevent_seqnum);


static ssize_t uevent_helper_show(struct kobject *kobj,
      struct kobj_attribute *attr, char *buf)
{
 return sprintf(buf, "%s\n", uevent_helper);
}
static ssize_t uevent_helper_store(struct kobject *kobj,
       struct kobj_attribute *attr,
       const char *buf, size_t count)
{
 if (count+1 > UEVENT_HELPER_PATH_LEN)
  return -ENOENT;
 memcpy(uevent_helper, buf, count);
 uevent_helper[count] = '\0';
 if (count && uevent_helper[count-1] == '\n')
  uevent_helper[count-1] = '\0';
 return count;
}
KERNEL_ATTR_RW(uevent_helper);

static ssize_t profiling_show(struct kobject *kobj,
      struct kobj_attribute *attr, char *buf)
{
 return sprintf(buf, "%d\n", prof_on);
}
static ssize_t profiling_store(struct kobject *kobj,
       struct kobj_attribute *attr,
       const char *buf, size_t count)
{
 int ret;

 if (prof_on)
  return -EEXIST;





 profile_setup((char *)buf);
 ret = profile_init();
 if (ret)
  return ret;
 ret = create_proc_profile();
 if (ret)
  return ret;
 return count;
}
KERNEL_ATTR_RW(profiling);

static ssize_t kexec_loaded_show(struct kobject *kobj,
     struct kobj_attribute *attr, char *buf)
{
 return sprintf(buf, "%d\n", !!kexec_image);
}
KERNEL_ATTR_RO(kexec_loaded);

static ssize_t kexec_crash_loaded_show(struct kobject *kobj,
           struct kobj_attribute *attr, char *buf)
{
 return sprintf(buf, "%d\n", !!kexec_crash_image);
}
KERNEL_ATTR_RO(kexec_crash_loaded);

static ssize_t kexec_crash_size_show(struct kobject *kobj,
           struct kobj_attribute *attr, char *buf)
{
 return sprintf(buf, "%zu\n", crash_get_memory_size());
}
static ssize_t kexec_crash_size_store(struct kobject *kobj,
       struct kobj_attribute *attr,
       const char *buf, size_t count)
{
 unsigned long cnt;
 int ret;

 if (kstrtoul(buf, 0, &cnt))
  return -EINVAL;

 ret = crash_shrink_memory(cnt);
 return ret < 0 ? ret : count;
}
KERNEL_ATTR_RW(kexec_crash_size);

static ssize_t vmcoreinfo_show(struct kobject *kobj,
          struct kobj_attribute *attr, char *buf)
{
 return sprintf(buf, "%lx %x\n",
         paddr_vmcoreinfo_note(),
         (unsigned int)sizeof(vmcoreinfo_note));
}
KERNEL_ATTR_RO(vmcoreinfo);



static ssize_t fscaps_show(struct kobject *kobj,
      struct kobj_attribute *attr, char *buf)
{
 return sprintf(buf, "%d\n", file_caps_enabled);
}
KERNEL_ATTR_RO(fscaps);

int rcu_expedited;
static ssize_t rcu_expedited_show(struct kobject *kobj,
      struct kobj_attribute *attr, char *buf)
{
 return sprintf(buf, "%d\n", READ_ONCE(rcu_expedited));
}
static ssize_t rcu_expedited_store(struct kobject *kobj,
       struct kobj_attribute *attr,
       const char *buf, size_t count)
{
 if (kstrtoint(buf, 0, &rcu_expedited))
  return -EINVAL;

 return count;
}
KERNEL_ATTR_RW(rcu_expedited);

int rcu_normal;
static ssize_t rcu_normal_show(struct kobject *kobj,
          struct kobj_attribute *attr, char *buf)
{
 return sprintf(buf, "%d\n", READ_ONCE(rcu_normal));
}
static ssize_t rcu_normal_store(struct kobject *kobj,
    struct kobj_attribute *attr,
    const char *buf, size_t count)
{
 if (kstrtoint(buf, 0, &rcu_normal))
  return -EINVAL;

 return count;
}
KERNEL_ATTR_RW(rcu_normal);




extern const void __start_notes __weak;
extern const void __stop_notes __weak;

static ssize_t notes_read(struct file *filp, struct kobject *kobj,
     struct bin_attribute *bin_attr,
     char *buf, loff_t off, size_t count)
{
 memcpy(buf, &__start_notes + off, count);
 return count;
}

static struct bin_attribute notes_attr = {
 .attr = {
  .name = "notes",
  .mode = S_IRUGO,
 },
 .read = &notes_read,
};

struct kobject *kernel_kobj;
EXPORT_SYMBOL_GPL(kernel_kobj);

static struct attribute * kernel_attrs[] = {
 &fscaps_attr.attr,
 &uevent_seqnum_attr.attr,
 &uevent_helper_attr.attr,
 &profiling_attr.attr,
 &kexec_loaded_attr.attr,
 &kexec_crash_loaded_attr.attr,
 &kexec_crash_size_attr.attr,
 &vmcoreinfo_attr.attr,
 &rcu_expedited_attr.attr,
 &rcu_normal_attr.attr,
 NULL
};

static struct attribute_group kernel_attr_group = {
 .attrs = kernel_attrs,
};

static int __init ksysfs_init(void)
{
 int error;

 kernel_kobj = kobject_create_and_add("kernel", NULL);
 if (!kernel_kobj) {
  error = -ENOMEM;
  goto exit;
 }
 error = sysfs_create_group(kernel_kobj, &kernel_attr_group);
 if (error)
  goto kset_exit;

 if (notes_size > 0) {
  notes_attr.size = notes_size;
  error = sysfs_create_bin_file(kernel_kobj, &notes_attr);
  if (error)
   goto group_exit;
 }

 return 0;

group_exit:
 sysfs_remove_group(kernel_kobj, &kernel_attr_group);
kset_exit:
 kobject_put(kernel_kobj);
exit:
 return error;
}

core_initcall(ksysfs_init);








static DEFINE_SPINLOCK(kthread_create_lock);
static LIST_HEAD(kthread_create_list);
struct task_struct *kthreadd_task;

struct kthread_create_info
{

 int (*threadfn)(void *data);
 void *data;
 int node;


 struct task_struct *result;
 struct completion *done;

 struct list_head list;
};

struct kthread {
 unsigned long flags;
 unsigned int cpu;
 void *data;
 struct completion parked;
 struct completion exited;
};

enum KTHREAD_BITS {
 KTHREAD_IS_PER_CPU = 0,
 KTHREAD_SHOULD_STOP,
 KTHREAD_SHOULD_PARK,
 KTHREAD_IS_PARKED,
};

 container_of(vfork, struct kthread, exited)

static inline struct kthread *to_kthread(struct task_struct *k)
{
 return __to_kthread(k->vfork_done);
}

static struct kthread *to_live_kthread(struct task_struct *k)
{
 struct completion *vfork = ACCESS_ONCE(k->vfork_done);
 if (likely(vfork))
  return __to_kthread(vfork);
 return NULL;
}
bool kthread_should_stop(void)
{
 return test_bit(KTHREAD_SHOULD_STOP, &to_kthread(current)->flags);
}
EXPORT_SYMBOL(kthread_should_stop);
bool kthread_should_park(void)
{
 return test_bit(KTHREAD_SHOULD_PARK, &to_kthread(current)->flags);
}
EXPORT_SYMBOL_GPL(kthread_should_park);
bool kthread_freezable_should_stop(bool *was_frozen)
{
 bool frozen = false;

 might_sleep();

 if (unlikely(freezing(current)))
  frozen = __refrigerator(true);

 if (was_frozen)
  *was_frozen = frozen;

 return kthread_should_stop();
}
EXPORT_SYMBOL_GPL(kthread_freezable_should_stop);
void *kthread_data(struct task_struct *task)
{
 return to_kthread(task)->data;
}
void *probe_kthread_data(struct task_struct *task)
{
 struct kthread *kthread = to_kthread(task);
 void *data = NULL;

 probe_kernel_read(&data, &kthread->data, sizeof(data));
 return data;
}

static void __kthread_parkme(struct kthread *self)
{
 __set_current_state(TASK_PARKED);
 while (test_bit(KTHREAD_SHOULD_PARK, &self->flags)) {
  if (!test_and_set_bit(KTHREAD_IS_PARKED, &self->flags))
   complete(&self->parked);
  schedule();
  __set_current_state(TASK_PARKED);
 }
 clear_bit(KTHREAD_IS_PARKED, &self->flags);
 __set_current_state(TASK_RUNNING);
}

void kthread_parkme(void)
{
 __kthread_parkme(to_kthread(current));
}
EXPORT_SYMBOL_GPL(kthread_parkme);

static int kthread(void *_create)
{

 struct kthread_create_info *create = _create;
 int (*threadfn)(void *data) = create->threadfn;
 void *data = create->data;
 struct completion *done;
 struct kthread self;
 int ret;

 self.flags = 0;
 self.data = data;
 init_completion(&self.exited);
 init_completion(&self.parked);
 current->vfork_done = &self.exited;


 done = xchg(&create->done, NULL);
 if (!done) {
  kfree(create);
  do_exit(-EINTR);
 }

 __set_current_state(TASK_UNINTERRUPTIBLE);
 create->result = current;
 complete(done);
 schedule();

 ret = -EINTR;

 if (!test_bit(KTHREAD_SHOULD_STOP, &self.flags)) {
  __kthread_parkme(&self);
  ret = threadfn(data);
 }

 do_exit(ret);
}


int tsk_fork_get_node(struct task_struct *tsk)
{
 if (tsk == kthreadd_task)
  return tsk->pref_node_fork;
 return NUMA_NO_NODE;
}

static void create_kthread(struct kthread_create_info *create)
{
 int pid;

 current->pref_node_fork = create->node;

 pid = kernel_thread(kthread, create, CLONE_FS | CLONE_FILES | SIGCHLD);
 if (pid < 0) {

  struct completion *done = xchg(&create->done, NULL);

  if (!done) {
   kfree(create);
   return;
  }
  create->result = ERR_PTR(pid);
  complete(done);
 }
}
struct task_struct *kthread_create_on_node(int (*threadfn)(void *data),
        void *data, int node,
        const char namefmt[],
        ...)
{
 DECLARE_COMPLETION_ONSTACK(done);
 struct task_struct *task;
 struct kthread_create_info *create = kmalloc(sizeof(*create),
           GFP_KERNEL);

 if (!create)
  return ERR_PTR(-ENOMEM);
 create->threadfn = threadfn;
 create->data = data;
 create->node = node;
 create->done = &done;

 spin_lock(&kthread_create_lock);
 list_add_tail(&create->list, &kthread_create_list);
 spin_unlock(&kthread_create_lock);

 wake_up_process(kthreadd_task);





 if (unlikely(wait_for_completion_killable(&done))) {





  if (xchg(&create->done, NULL))
   return ERR_PTR(-EINTR);




  wait_for_completion(&done);
 }
 task = create->result;
 if (!IS_ERR(task)) {
  static const struct sched_param param = { .sched_priority = 0 };
  va_list args;

  va_start(args, namefmt);
  vsnprintf(task->comm, sizeof(task->comm), namefmt, args);
  va_end(args);




  sched_setscheduler_nocheck(task, SCHED_NORMAL, &param);
  set_cpus_allowed_ptr(task, cpu_all_mask);
 }
 kfree(create);
 return task;
}
EXPORT_SYMBOL(kthread_create_on_node);

static void __kthread_bind_mask(struct task_struct *p, const struct cpumask *mask, long state)
{
 unsigned long flags;

 if (!wait_task_inactive(p, state)) {
  WARN_ON(1);
  return;
 }


 raw_spin_lock_irqsave(&p->pi_lock, flags);
 do_set_cpus_allowed(p, mask);
 p->flags |= PF_NO_SETAFFINITY;
 raw_spin_unlock_irqrestore(&p->pi_lock, flags);
}

static void __kthread_bind(struct task_struct *p, unsigned int cpu, long state)
{
 __kthread_bind_mask(p, cpumask_of(cpu), state);
}

void kthread_bind_mask(struct task_struct *p, const struct cpumask *mask)
{
 __kthread_bind_mask(p, mask, TASK_UNINTERRUPTIBLE);
}
void kthread_bind(struct task_struct *p, unsigned int cpu)
{
 __kthread_bind(p, cpu, TASK_UNINTERRUPTIBLE);
}
EXPORT_SYMBOL(kthread_bind);
struct task_struct *kthread_create_on_cpu(int (*threadfn)(void *data),
       void *data, unsigned int cpu,
       const char *namefmt)
{
 struct task_struct *p;

 p = kthread_create_on_node(threadfn, data, cpu_to_node(cpu), namefmt,
       cpu);
 if (IS_ERR(p))
  return p;
 set_bit(KTHREAD_IS_PER_CPU, &to_kthread(p)->flags);
 to_kthread(p)->cpu = cpu;

 kthread_park(p);
 return p;
}

static void __kthread_unpark(struct task_struct *k, struct kthread *kthread)
{
 clear_bit(KTHREAD_SHOULD_PARK, &kthread->flags);






 if (test_and_clear_bit(KTHREAD_IS_PARKED, &kthread->flags)) {
  if (test_bit(KTHREAD_IS_PER_CPU, &kthread->flags))
   __kthread_bind(k, kthread->cpu, TASK_PARKED);
  wake_up_state(k, TASK_PARKED);
 }
}
void kthread_unpark(struct task_struct *k)
{
 struct kthread *kthread = to_live_kthread(k);

 if (kthread)
  __kthread_unpark(k, kthread);
}
EXPORT_SYMBOL_GPL(kthread_unpark);
int kthread_park(struct task_struct *k)
{
 struct kthread *kthread = to_live_kthread(k);
 int ret = -ENOSYS;

 if (kthread) {
  if (!test_bit(KTHREAD_IS_PARKED, &kthread->flags)) {
   set_bit(KTHREAD_SHOULD_PARK, &kthread->flags);
   if (k != current) {
    wake_up_process(k);
    wait_for_completion(&kthread->parked);
   }
  }
  ret = 0;
 }
 return ret;
}
EXPORT_SYMBOL_GPL(kthread_park);
int kthread_stop(struct task_struct *k)
{
 struct kthread *kthread;
 int ret;

 trace_sched_kthread_stop(k);

 get_task_struct(k);
 kthread = to_live_kthread(k);
 if (kthread) {
  set_bit(KTHREAD_SHOULD_STOP, &kthread->flags);
  __kthread_unpark(k, kthread);
  wake_up_process(k);
  wait_for_completion(&kthread->exited);
 }
 ret = k->exit_code;
 put_task_struct(k);

 trace_sched_kthread_stop_ret(ret);
 return ret;
}
EXPORT_SYMBOL(kthread_stop);

int kthreadd(void *unused)
{
 struct task_struct *tsk = current;


 set_task_comm(tsk, "kthreadd");
 ignore_signals(tsk);
 set_cpus_allowed_ptr(tsk, cpu_all_mask);
 set_mems_allowed(node_states[N_MEMORY]);

 current->flags |= PF_NOFREEZE;

 for (;;) {
  set_current_state(TASK_INTERRUPTIBLE);
  if (list_empty(&kthread_create_list))
   schedule();
  __set_current_state(TASK_RUNNING);

  spin_lock(&kthread_create_lock);
  while (!list_empty(&kthread_create_list)) {
   struct kthread_create_info *create;

   create = list_entry(kthread_create_list.next,
         struct kthread_create_info, list);
   list_del_init(&create->list);
   spin_unlock(&kthread_create_lock);

   create_kthread(create);

   spin_lock(&kthread_create_lock);
  }
  spin_unlock(&kthread_create_lock);
 }

 return 0;
}

void __init_kthread_worker(struct kthread_worker *worker,
    const char *name,
    struct lock_class_key *key)
{
 spin_lock_init(&worker->lock);
 lockdep_set_class_and_name(&worker->lock, key, name);
 INIT_LIST_HEAD(&worker->work_list);
 worker->task = NULL;
}
EXPORT_SYMBOL_GPL(__init_kthread_worker);
int kthread_worker_fn(void *worker_ptr)
{
 struct kthread_worker *worker = worker_ptr;
 struct kthread_work *work;

 WARN_ON(worker->task);
 worker->task = current;
repeat:
 set_current_state(TASK_INTERRUPTIBLE);

 if (kthread_should_stop()) {
  __set_current_state(TASK_RUNNING);
  spin_lock_irq(&worker->lock);
  worker->task = NULL;
  spin_unlock_irq(&worker->lock);
  return 0;
 }

 work = NULL;
 spin_lock_irq(&worker->lock);
 if (!list_empty(&worker->work_list)) {
  work = list_first_entry(&worker->work_list,
     struct kthread_work, node);
  list_del_init(&work->node);
 }
 worker->current_work = work;
 spin_unlock_irq(&worker->lock);

 if (work) {
  __set_current_state(TASK_RUNNING);
  work->func(work);
 } else if (!freezing(current))
  schedule();

 try_to_freeze();
 goto repeat;
}
EXPORT_SYMBOL_GPL(kthread_worker_fn);


static void insert_kthread_work(struct kthread_worker *worker,
          struct kthread_work *work,
          struct list_head *pos)
{
 lockdep_assert_held(&worker->lock);

 list_add_tail(&work->node, pos);
 work->worker = worker;
 if (!worker->current_work && likely(worker->task))
  wake_up_process(worker->task);
}
bool queue_kthread_work(struct kthread_worker *worker,
   struct kthread_work *work)
{
 bool ret = false;
 unsigned long flags;

 spin_lock_irqsave(&worker->lock, flags);
 if (list_empty(&work->node)) {
  insert_kthread_work(worker, work, &worker->work_list);
  ret = true;
 }
 spin_unlock_irqrestore(&worker->lock, flags);
 return ret;
}
EXPORT_SYMBOL_GPL(queue_kthread_work);

struct kthread_flush_work {
 struct kthread_work work;
 struct completion done;
};

static void kthread_flush_work_fn(struct kthread_work *work)
{
 struct kthread_flush_work *fwork =
  container_of(work, struct kthread_flush_work, work);
 complete(&fwork->done);
}







void flush_kthread_work(struct kthread_work *work)
{
 struct kthread_flush_work fwork = {
  KTHREAD_WORK_INIT(fwork.work, kthread_flush_work_fn),
  COMPLETION_INITIALIZER_ONSTACK(fwork.done),
 };
 struct kthread_worker *worker;
 bool noop = false;

retry:
 worker = work->worker;
 if (!worker)
  return;

 spin_lock_irq(&worker->lock);
 if (work->worker != worker) {
  spin_unlock_irq(&worker->lock);
  goto retry;
 }

 if (!list_empty(&work->node))
  insert_kthread_work(worker, &fwork.work, work->node.next);
 else if (worker->current_work == work)
  insert_kthread_work(worker, &fwork.work, worker->work_list.next);
 else
  noop = true;

 spin_unlock_irq(&worker->lock);

 if (!noop)
  wait_for_completion(&fwork.done);
}
EXPORT_SYMBOL_GPL(flush_kthread_work);
void flush_kthread_worker(struct kthread_worker *worker)
{
 struct kthread_flush_work fwork = {
  KTHREAD_WORK_INIT(fwork.work, kthread_flush_work_fn),
  COMPLETION_INITIALIZER_ONSTACK(fwork.done),
 };

 queue_kthread_work(worker, &fwork.work);
 wait_for_completion(&fwork.done);
}
EXPORT_SYMBOL_GPL(flush_kthread_worker);

static DEFINE_RAW_SPINLOCK(latency_lock);

static struct latency_record latency_record[MAXLR];

int latencytop_enabled;

void clear_all_latency_tracing(struct task_struct *p)
{
 unsigned long flags;

 if (!latencytop_enabled)
  return;

 raw_spin_lock_irqsave(&latency_lock, flags);
 memset(&p->latency_record, 0, sizeof(p->latency_record));
 p->latency_record_count = 0;
 raw_spin_unlock_irqrestore(&latency_lock, flags);
}

static void clear_global_latency_tracing(void)
{
 unsigned long flags;

 raw_spin_lock_irqsave(&latency_lock, flags);
 memset(&latency_record, 0, sizeof(latency_record));
 raw_spin_unlock_irqrestore(&latency_lock, flags);
}

static void __sched
account_global_scheduler_latency(struct task_struct *tsk,
     struct latency_record *lat)
{
 int firstnonnull = MAXLR + 1;
 int i;

 if (!latencytop_enabled)
  return;


 if (!tsk->mm)
  return;

 for (i = 0; i < MAXLR; i++) {
  int q, same = 1;


  if (!latency_record[i].backtrace[0]) {
   if (firstnonnull > i)
    firstnonnull = i;
   continue;
  }
  for (q = 0; q < LT_BACKTRACEDEPTH; q++) {
   unsigned long record = lat->backtrace[q];

   if (latency_record[i].backtrace[q] != record) {
    same = 0;
    break;
   }


   if (record == 0 || record == ULONG_MAX)
    break;
  }
  if (same) {
   latency_record[i].count++;
   latency_record[i].time += lat->time;
   if (lat->time > latency_record[i].max)
    latency_record[i].max = lat->time;
   return;
  }
 }

 i = firstnonnull;
 if (i >= MAXLR - 1)
  return;


 memcpy(&latency_record[i], lat, sizeof(struct latency_record));
}




static inline void store_stacktrace(struct task_struct *tsk,
     struct latency_record *lat)
{
 struct stack_trace trace;

 memset(&trace, 0, sizeof(trace));
 trace.max_entries = LT_BACKTRACEDEPTH;
 trace.entries = &lat->backtrace[0];
 save_stack_trace_tsk(tsk, &trace);
}
void __sched
__account_scheduler_latency(struct task_struct *tsk, int usecs, int inter)
{
 unsigned long flags;
 int i, q;
 struct latency_record lat;


 if (inter && usecs > 5000)
  return;



 if (usecs <= 0)
  return;

 memset(&lat, 0, sizeof(lat));
 lat.count = 1;
 lat.time = usecs;
 lat.max = usecs;
 store_stacktrace(tsk, &lat);

 raw_spin_lock_irqsave(&latency_lock, flags);

 account_global_scheduler_latency(tsk, &lat);

 for (i = 0; i < tsk->latency_record_count; i++) {
  struct latency_record *mylat;
  int same = 1;

  mylat = &tsk->latency_record[i];
  for (q = 0; q < LT_BACKTRACEDEPTH; q++) {
   unsigned long record = lat.backtrace[q];

   if (mylat->backtrace[q] != record) {
    same = 0;
    break;
   }


   if (record == 0 || record == ULONG_MAX)
    break;
  }
  if (same) {
   mylat->count++;
   mylat->time += lat.time;
   if (lat.time > mylat->max)
    mylat->max = lat.time;
   goto out_unlock;
  }
 }




 if (tsk->latency_record_count >= LT_SAVECOUNT)
  goto out_unlock;


 i = tsk->latency_record_count++;
 memcpy(&tsk->latency_record[i], &lat, sizeof(struct latency_record));

out_unlock:
 raw_spin_unlock_irqrestore(&latency_lock, flags);
}

static int lstats_show(struct seq_file *m, void *v)
{
 int i;

 seq_puts(m, "Latency Top version : v0.1\n");

 for (i = 0; i < MAXLR; i++) {
  struct latency_record *lr = &latency_record[i];

  if (lr->backtrace[0]) {
   int q;
   seq_printf(m, "%i %lu %lu",
       lr->count, lr->time, lr->max);
   for (q = 0; q < LT_BACKTRACEDEPTH; q++) {
    unsigned long bt = lr->backtrace[q];
    if (!bt)
     break;
    if (bt == ULONG_MAX)
     break;
    seq_printf(m, " %ps", (void *)bt);
   }
   seq_puts(m, "\n");
  }
 }
 return 0;
}

static ssize_t
lstats_write(struct file *file, const char __user *buf, size_t count,
      loff_t *offs)
{
 clear_global_latency_tracing();

 return count;
}

static int lstats_open(struct inode *inode, struct file *filp)
{
 return single_open(filp, lstats_show, NULL);
}

static const struct file_operations lstats_fops = {
 .open = lstats_open,
 .read = seq_read,
 .write = lstats_write,
 .llseek = seq_lseek,
 .release = single_release,
};

static int __init init_lstats_procfs(void)
{
 proc_create("latency_stats", 0644, NULL, &lstats_fops);
 return 0;
}

int sysctl_latencytop(struct ctl_table *table, int write,
   void __user *buffer, size_t *lenp, loff_t *ppos)
{
 int err;

 err = proc_dointvec(table, write, buffer, lenp, ppos);
 if (latencytop_enabled)
  force_schedstat_enabled();

 return err;
}
device_initcall(init_lstats_procfs);


DEFINE_MUTEX(pm_mutex);




static BLOCKING_NOTIFIER_HEAD(pm_chain_head);

int register_pm_notifier(struct notifier_block *nb)
{
 return blocking_notifier_chain_register(&pm_chain_head, nb);
}
EXPORT_SYMBOL_GPL(register_pm_notifier);

int unregister_pm_notifier(struct notifier_block *nb)
{
 return blocking_notifier_chain_unregister(&pm_chain_head, nb);
}
EXPORT_SYMBOL_GPL(unregister_pm_notifier);

int pm_notifier_call_chain(unsigned long val)
{
 int ret = blocking_notifier_call_chain(&pm_chain_head, val, NULL);

 return notifier_to_errno(ret);
}


int pm_async_enabled = 1;

static ssize_t pm_async_show(struct kobject *kobj, struct kobj_attribute *attr,
        char *buf)
{
 return sprintf(buf, "%d\n", pm_async_enabled);
}

static ssize_t pm_async_store(struct kobject *kobj, struct kobj_attribute *attr,
         const char *buf, size_t n)
{
 unsigned long val;

 if (kstrtoul(buf, 10, &val))
  return -EINVAL;

 if (val > 1)
  return -EINVAL;

 pm_async_enabled = val;
 return n;
}

power_attr(pm_async);

int pm_test_level = TEST_NONE;

static const char * const pm_tests[__TEST_AFTER_LAST] = {
 [TEST_NONE] = "none",
 [TEST_CORE] = "core",
 [TEST_CPUS] = "processors",
 [TEST_PLATFORM] = "platform",
 [TEST_DEVICES] = "devices",
 [TEST_FREEZER] = "freezer",
};

static ssize_t pm_test_show(struct kobject *kobj, struct kobj_attribute *attr,
    char *buf)
{
 char *s = buf;
 int level;

 for (level = TEST_FIRST; level <= TEST_MAX; level++)
  if (pm_tests[level]) {
   if (level == pm_test_level)
    s += sprintf(s, "[%s] ", pm_tests[level]);
   else
    s += sprintf(s, "%s ", pm_tests[level]);
  }

 if (s != buf)

  *(s-1) = '\n';

 return (s - buf);
}

static ssize_t pm_test_store(struct kobject *kobj, struct kobj_attribute *attr,
    const char *buf, size_t n)
{
 const char * const *s;
 int level;
 char *p;
 int len;
 int error = -EINVAL;

 p = memchr(buf, '\n', n);
 len = p ? p - buf : n;

 lock_system_sleep();

 level = TEST_FIRST;
 for (s = &pm_tests[level]; level <= TEST_MAX; s++, level++)
  if (*s && len == strlen(*s) && !strncmp(buf, *s, len)) {
   pm_test_level = level;
   error = 0;
   break;
  }

 unlock_system_sleep();

 return error ? error : n;
}

power_attr(pm_test);

static char *suspend_step_name(enum suspend_stat_step step)
{
 switch (step) {
 case SUSPEND_FREEZE:
  return "freeze";
 case SUSPEND_PREPARE:
  return "prepare";
 case SUSPEND_SUSPEND:
  return "suspend";
 case SUSPEND_SUSPEND_NOIRQ:
  return "suspend_noirq";
 case SUSPEND_RESUME_NOIRQ:
  return "resume_noirq";
 case SUSPEND_RESUME:
  return "resume";
 default:
  return "";
 }
}

static int suspend_stats_show(struct seq_file *s, void *unused)
{
 int i, index, last_dev, last_errno, last_step;

 last_dev = suspend_stats.last_failed_dev + REC_FAILED_NUM - 1;
 last_dev %= REC_FAILED_NUM;
 last_errno = suspend_stats.last_failed_errno + REC_FAILED_NUM - 1;
 last_errno %= REC_FAILED_NUM;
 last_step = suspend_stats.last_failed_step + REC_FAILED_NUM - 1;
 last_step %= REC_FAILED_NUM;
 seq_printf(s, "%s: %d\n%s: %d\n%s: %d\n%s: %d\n%s: %d\n"
   "%s: %d\n%s: %d\n%s: %d\n%s: %d\n%s: %d\n",
   "success", suspend_stats.success,
   "fail", suspend_stats.fail,
   "failed_freeze", suspend_stats.failed_freeze,
   "failed_prepare", suspend_stats.failed_prepare,
   "failed_suspend", suspend_stats.failed_suspend,
   "failed_suspend_late",
    suspend_stats.failed_suspend_late,
   "failed_suspend_noirq",
    suspend_stats.failed_suspend_noirq,
   "failed_resume", suspend_stats.failed_resume,
   "failed_resume_early",
    suspend_stats.failed_resume_early,
   "failed_resume_noirq",
    suspend_stats.failed_resume_noirq);
 seq_printf(s, "failures:\n  last_failed_dev:\t%-s\n",
   suspend_stats.failed_devs[last_dev]);
 for (i = 1; i < REC_FAILED_NUM; i++) {
  index = last_dev + REC_FAILED_NUM - i;
  index %= REC_FAILED_NUM;
  seq_printf(s, "\t\t\t%-s\n",
   suspend_stats.failed_devs[index]);
 }
 seq_printf(s, "  last_failed_errno:\t%-d\n",
   suspend_stats.errno[last_errno]);
 for (i = 1; i < REC_FAILED_NUM; i++) {
  index = last_errno + REC_FAILED_NUM - i;
  index %= REC_FAILED_NUM;
  seq_printf(s, "\t\t\t%-d\n",
   suspend_stats.errno[index]);
 }
 seq_printf(s, "  last_failed_step:\t%-s\n",
   suspend_step_name(
    suspend_stats.failed_steps[last_step]));
 for (i = 1; i < REC_FAILED_NUM; i++) {
  index = last_step + REC_FAILED_NUM - i;
  index %= REC_FAILED_NUM;
  seq_printf(s, "\t\t\t%-s\n",
   suspend_step_name(
    suspend_stats.failed_steps[index]));
 }

 return 0;
}

static int suspend_stats_open(struct inode *inode, struct file *file)
{
 return single_open(file, suspend_stats_show, NULL);
}

static const struct file_operations suspend_stats_operations = {
 .open = suspend_stats_open,
 .read = seq_read,
 .llseek = seq_lseek,
 .release = single_release,
};

static int __init pm_debugfs_init(void)
{
 debugfs_create_file("suspend_stats", S_IFREG | S_IRUGO,
   NULL, NULL, &suspend_stats_operations);
 return 0;
}

late_initcall(pm_debugfs_init);








bool pm_print_times_enabled;

static ssize_t pm_print_times_show(struct kobject *kobj,
       struct kobj_attribute *attr, char *buf)
{
 return sprintf(buf, "%d\n", pm_print_times_enabled);
}

static ssize_t pm_print_times_store(struct kobject *kobj,
        struct kobj_attribute *attr,
        const char *buf, size_t n)
{
 unsigned long val;

 if (kstrtoul(buf, 10, &val))
  return -EINVAL;

 if (val > 1)
  return -EINVAL;

 pm_print_times_enabled = !!val;
 return n;
}

power_attr(pm_print_times);

static inline void pm_print_times_init(void)
{
 pm_print_times_enabled = !!initcall_debug;
}

static ssize_t pm_wakeup_irq_show(struct kobject *kobj,
     struct kobj_attribute *attr,
     char *buf)
{
 return pm_wakeup_irq ? sprintf(buf, "%u\n", pm_wakeup_irq) : -ENODATA;
}

power_attr_ro(pm_wakeup_irq);

static inline void pm_print_times_init(void) {}

struct kobject *power_kobj;
static ssize_t state_show(struct kobject *kobj, struct kobj_attribute *attr,
     char *buf)
{
 char *s = buf;
 suspend_state_t i;

 for (i = PM_SUSPEND_MIN; i < PM_SUSPEND_MAX; i++)
  if (pm_states[i])
   s += sprintf(s,"%s ", pm_states[i]);

 if (hibernation_available())
  s += sprintf(s, "disk ");
 if (s != buf)

  *(s-1) = '\n';
 return (s - buf);
}

static suspend_state_t decode_state(const char *buf, size_t n)
{
 suspend_state_t state;
 char *p;
 int len;

 p = memchr(buf, '\n', n);
 len = p ? p - buf : n;


 if (len == 4 && !strncmp(buf, "disk", len))
  return PM_SUSPEND_MAX;

 for (state = PM_SUSPEND_MIN; state < PM_SUSPEND_MAX; state++) {
  const char *label = pm_states[state];

  if (label && len == strlen(label) && !strncmp(buf, label, len))
   return state;
 }

 return PM_SUSPEND_ON;
}

static ssize_t state_store(struct kobject *kobj, struct kobj_attribute *attr,
      const char *buf, size_t n)
{
 suspend_state_t state;
 int error;

 error = pm_autosleep_lock();
 if (error)
  return error;

 if (pm_autosleep_state() > PM_SUSPEND_ON) {
  error = -EBUSY;
  goto out;
 }

 state = decode_state(buf, n);
 if (state < PM_SUSPEND_MAX)
  error = pm_suspend(state);
 else if (state == PM_SUSPEND_MAX)
  error = hibernate();
 else
  error = -EINVAL;

 out:
 pm_autosleep_unlock();
 return error ? error : n;
}

power_attr(state);

static ssize_t wakeup_count_show(struct kobject *kobj,
    struct kobj_attribute *attr,
    char *buf)
{
 unsigned int val;

 return pm_get_wakeup_count(&val, true) ?
  sprintf(buf, "%u\n", val) : -EINTR;
}

static ssize_t wakeup_count_store(struct kobject *kobj,
    struct kobj_attribute *attr,
    const char *buf, size_t n)
{
 unsigned int val;
 int error;

 error = pm_autosleep_lock();
 if (error)
  return error;

 if (pm_autosleep_state() > PM_SUSPEND_ON) {
  error = -EBUSY;
  goto out;
 }

 error = -EINVAL;
 if (sscanf(buf, "%u", &val) == 1) {
  if (pm_save_wakeup_count(val))
   error = n;
  else
   pm_print_active_wakeup_sources();
 }

 out:
 pm_autosleep_unlock();
 return error;
}

power_attr(wakeup_count);

static ssize_t autosleep_show(struct kobject *kobj,
         struct kobj_attribute *attr,
         char *buf)
{
 suspend_state_t state = pm_autosleep_state();

 if (state == PM_SUSPEND_ON)
  return sprintf(buf, "off\n");

 if (state < PM_SUSPEND_MAX)
  return sprintf(buf, "%s\n", pm_states[state] ?
     pm_states[state] : "error");
 return sprintf(buf, "disk\n");
 return sprintf(buf, "error");
}

static ssize_t autosleep_store(struct kobject *kobj,
          struct kobj_attribute *attr,
          const char *buf, size_t n)
{
 suspend_state_t state = decode_state(buf, n);
 int error;

 if (state == PM_SUSPEND_ON
     && strcmp(buf, "off") && strcmp(buf, "off\n"))
  return -EINVAL;

 error = pm_autosleep_set_state(state);
 return error ? error : n;
}

power_attr(autosleep);

static ssize_t wake_lock_show(struct kobject *kobj,
         struct kobj_attribute *attr,
         char *buf)
{
 return pm_show_wakelocks(buf, true);
}

static ssize_t wake_lock_store(struct kobject *kobj,
          struct kobj_attribute *attr,
          const char *buf, size_t n)
{
 int error = pm_wake_lock(buf);
 return error ? error : n;
}

power_attr(wake_lock);

static ssize_t wake_unlock_show(struct kobject *kobj,
    struct kobj_attribute *attr,
    char *buf)
{
 return pm_show_wakelocks(buf, false);
}

static ssize_t wake_unlock_store(struct kobject *kobj,
     struct kobj_attribute *attr,
     const char *buf, size_t n)
{
 int error = pm_wake_unlock(buf);
 return error ? error : n;
}

power_attr(wake_unlock);


int pm_trace_enabled;

static ssize_t pm_trace_show(struct kobject *kobj, struct kobj_attribute *attr,
        char *buf)
{
 return sprintf(buf, "%d\n", pm_trace_enabled);
}

static ssize_t
pm_trace_store(struct kobject *kobj, struct kobj_attribute *attr,
        const char *buf, size_t n)
{
 int val;

 if (sscanf(buf, "%d", &val) == 1) {
  pm_trace_enabled = !!val;
  if (pm_trace_enabled) {
   pr_warn("PM: Enabling pm_trace changes system date and time during resume.\n"
    "PM: Correct system time has to be restored manually after resume.\n");
  }
  return n;
 }
 return -EINVAL;
}

power_attr(pm_trace);

static ssize_t pm_trace_dev_match_show(struct kobject *kobj,
           struct kobj_attribute *attr,
           char *buf)
{
 return show_trace_dev_match(buf, PAGE_SIZE);
}

power_attr_ro(pm_trace_dev_match);


static ssize_t pm_freeze_timeout_show(struct kobject *kobj,
          struct kobj_attribute *attr, char *buf)
{
 return sprintf(buf, "%u\n", freeze_timeout_msecs);
}

static ssize_t pm_freeze_timeout_store(struct kobject *kobj,
           struct kobj_attribute *attr,
           const char *buf, size_t n)
{
 unsigned long val;

 if (kstrtoul(buf, 10, &val))
  return -EINVAL;

 freeze_timeout_msecs = val;
 return n;
}

power_attr(pm_freeze_timeout);


static struct attribute * g[] = {
 &state_attr.attr,
 &pm_trace_attr.attr,
 &pm_trace_dev_match_attr.attr,
 &pm_async_attr.attr,
 &wakeup_count_attr.attr,
 &autosleep_attr.attr,
 &wake_lock_attr.attr,
 &wake_unlock_attr.attr,
 &pm_test_attr.attr,
 &pm_print_times_attr.attr,
 &pm_wakeup_irq_attr.attr,
 &pm_freeze_timeout_attr.attr,
 NULL,
};

static struct attribute_group attr_group = {
 .attrs = g,
};

struct workqueue_struct *pm_wq;
EXPORT_SYMBOL_GPL(pm_wq);

static int __init pm_start_workqueue(void)
{
 pm_wq = alloc_workqueue("pm", WQ_FREEZABLE, 0);

 return pm_wq ? 0 : -ENOMEM;
}

static int __init pm_init(void)
{
 int error = pm_start_workqueue();
 if (error)
  return error;
 hibernate_image_size_init();
 hibernate_reserved_size_init();
 power_kobj = kobject_create_and_add("power", NULL);
 if (!power_kobj)
  return -ENOMEM;
 error = sysfs_create_group(power_kobj, &attr_group);
 if (error)
  return error;
 pm_print_times_init();
 return pm_autosleep_init();
}

core_initcall(pm_init);



__read_mostly bool force_irqthreads;

static int __init setup_forced_irqthreads(char *arg)
{
 force_irqthreads = true;
 return 0;
}
early_param("threadirqs", setup_forced_irqthreads);

static void __synchronize_hardirq(struct irq_desc *desc)
{
 bool inprogress;

 do {
  unsigned long flags;





  while (irqd_irq_inprogress(&desc->irq_data))
   cpu_relax();


  raw_spin_lock_irqsave(&desc->lock, flags);
  inprogress = irqd_irq_inprogress(&desc->irq_data);
  raw_spin_unlock_irqrestore(&desc->lock, flags);


 } while (inprogress);
}
bool synchronize_hardirq(unsigned int irq)
{
 struct irq_desc *desc = irq_to_desc(irq);

 if (desc) {
  __synchronize_hardirq(desc);
  return !atomic_read(&desc->threads_active);
 }

 return true;
}
EXPORT_SYMBOL(synchronize_hardirq);
void synchronize_irq(unsigned int irq)
{
 struct irq_desc *desc = irq_to_desc(irq);

 if (desc) {
  __synchronize_hardirq(desc);





  wait_event(desc->wait_for_threads,
      !atomic_read(&desc->threads_active));
 }
}
EXPORT_SYMBOL(synchronize_irq);

cpumask_var_t irq_default_affinity;

static int __irq_can_set_affinity(struct irq_desc *desc)
{
 if (!desc || !irqd_can_balance(&desc->irq_data) ||
     !desc->irq_data.chip || !desc->irq_data.chip->irq_set_affinity)
  return 0;
 return 1;
}






int irq_can_set_affinity(unsigned int irq)
{
 return __irq_can_set_affinity(irq_to_desc(irq));
}
void irq_set_thread_affinity(struct irq_desc *desc)
{
 struct irqaction *action;

 for_each_action_of_desc(desc, action)
  if (action->thread)
   set_bit(IRQTF_AFFINITY, &action->thread_flags);
}

static inline bool irq_can_move_pcntxt(struct irq_data *data)
{
 return irqd_can_move_in_process_context(data);
}
static inline bool irq_move_pending(struct irq_data *data)
{
 return irqd_is_setaffinity_pending(data);
}
static inline void
irq_copy_pending(struct irq_desc *desc, const struct cpumask *mask)
{
 cpumask_copy(desc->pending_mask, mask);
}
static inline void
irq_get_pending(struct cpumask *mask, struct irq_desc *desc)
{
 cpumask_copy(mask, desc->pending_mask);
}
static inline bool irq_can_move_pcntxt(struct irq_data *data) { return true; }
static inline bool irq_move_pending(struct irq_data *data) { return false; }
static inline void
irq_copy_pending(struct irq_desc *desc, const struct cpumask *mask) { }
static inline void
irq_get_pending(struct cpumask *mask, struct irq_desc *desc) { }

int irq_do_set_affinity(struct irq_data *data, const struct cpumask *mask,
   bool force)
{
 struct irq_desc *desc = irq_data_to_desc(data);
 struct irq_chip *chip = irq_data_get_irq_chip(data);
 int ret;

 ret = chip->irq_set_affinity(data, mask, force);
 switch (ret) {
 case IRQ_SET_MASK_OK:
 case IRQ_SET_MASK_OK_DONE:
  cpumask_copy(desc->irq_common_data.affinity, mask);
 case IRQ_SET_MASK_OK_NOCOPY:
  irq_set_thread_affinity(desc);
  ret = 0;
 }

 return ret;
}

int irq_set_affinity_locked(struct irq_data *data, const struct cpumask *mask,
       bool force)
{
 struct irq_chip *chip = irq_data_get_irq_chip(data);
 struct irq_desc *desc = irq_data_to_desc(data);
 int ret = 0;

 if (!chip || !chip->irq_set_affinity)
  return -EINVAL;

 if (irq_can_move_pcntxt(data)) {
  ret = irq_do_set_affinity(data, mask, force);
 } else {
  irqd_set_move_pending(data);
  irq_copy_pending(desc, mask);
 }

 if (desc->affinity_notify) {
  kref_get(&desc->affinity_notify->kref);
  schedule_work(&desc->affinity_notify->work);
 }
 irqd_set(data, IRQD_AFFINITY_SET);

 return ret;
}

int __irq_set_affinity(unsigned int irq, const struct cpumask *mask, bool force)
{
 struct irq_desc *desc = irq_to_desc(irq);
 unsigned long flags;
 int ret;

 if (!desc)
  return -EINVAL;

 raw_spin_lock_irqsave(&desc->lock, flags);
 ret = irq_set_affinity_locked(irq_desc_get_irq_data(desc), mask, force);
 raw_spin_unlock_irqrestore(&desc->lock, flags);
 return ret;
}

int irq_set_affinity_hint(unsigned int irq, const struct cpumask *m)
{
 unsigned long flags;
 struct irq_desc *desc = irq_get_desc_lock(irq, &flags, IRQ_GET_DESC_CHECK_GLOBAL);

 if (!desc)
  return -EINVAL;
 desc->affinity_hint = m;
 irq_put_desc_unlock(desc, flags);

 if (m)
  __irq_set_affinity(irq, m, false);
 return 0;
}
EXPORT_SYMBOL_GPL(irq_set_affinity_hint);

static void irq_affinity_notify(struct work_struct *work)
{
 struct irq_affinity_notify *notify =
  container_of(work, struct irq_affinity_notify, work);
 struct irq_desc *desc = irq_to_desc(notify->irq);
 cpumask_var_t cpumask;
 unsigned long flags;

 if (!desc || !alloc_cpumask_var(&cpumask, GFP_KERNEL))
  goto out;

 raw_spin_lock_irqsave(&desc->lock, flags);
 if (irq_move_pending(&desc->irq_data))
  irq_get_pending(cpumask, desc);
 else
  cpumask_copy(cpumask, desc->irq_common_data.affinity);
 raw_spin_unlock_irqrestore(&desc->lock, flags);

 notify->notify(notify, cpumask);

 free_cpumask_var(cpumask);
out:
 kref_put(&notify->kref, notify->release);
}
int
irq_set_affinity_notifier(unsigned int irq, struct irq_affinity_notify *notify)
{
 struct irq_desc *desc = irq_to_desc(irq);
 struct irq_affinity_notify *old_notify;
 unsigned long flags;


 might_sleep();

 if (!desc)
  return -EINVAL;


 if (notify) {
  notify->irq = irq;
  kref_init(&notify->kref);
  INIT_WORK(&notify->work, irq_affinity_notify);
 }

 raw_spin_lock_irqsave(&desc->lock, flags);
 old_notify = desc->affinity_notify;
 desc->affinity_notify = notify;
 raw_spin_unlock_irqrestore(&desc->lock, flags);

 if (old_notify)
  kref_put(&old_notify->kref, old_notify->release);

 return 0;
}
EXPORT_SYMBOL_GPL(irq_set_affinity_notifier);




static int setup_affinity(struct irq_desc *desc, struct cpumask *mask)
{
 struct cpumask *set = irq_default_affinity;
 int node = irq_desc_get_node(desc);


 if (!__irq_can_set_affinity(desc))
  return 0;





 if (irqd_has_set(&desc->irq_data, IRQD_AFFINITY_SET)) {
  if (cpumask_intersects(desc->irq_common_data.affinity,
           cpu_online_mask))
   set = desc->irq_common_data.affinity;
  else
   irqd_clear(&desc->irq_data, IRQD_AFFINITY_SET);
 }

 cpumask_and(mask, cpu_online_mask, set);
 if (node != NUMA_NO_NODE) {
  const struct cpumask *nodemask = cpumask_of_node(node);


  if (cpumask_intersects(mask, nodemask))
   cpumask_and(mask, mask, nodemask);
 }
 irq_do_set_affinity(&desc->irq_data, mask, false);
 return 0;
}

static inline int setup_affinity(struct irq_desc *d, struct cpumask *mask)
{
 return irq_select_affinity(irq_desc_get_irq(d));
}




int irq_select_affinity_usr(unsigned int irq, struct cpumask *mask)
{
 struct irq_desc *desc = irq_to_desc(irq);
 unsigned long flags;
 int ret;

 raw_spin_lock_irqsave(&desc->lock, flags);
 ret = setup_affinity(desc, mask);
 raw_spin_unlock_irqrestore(&desc->lock, flags);
 return ret;
}

{
 return 0;
}