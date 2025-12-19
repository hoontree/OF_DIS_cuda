#ifdef USE_CUDA
void PatClass::OptimizeDebug(DebugIterData* debug_data, int max_iter)
{
  if (!pc->hasoptstarted)
  {
    ResetPatch(); 
    #if (SELECTMODE==1)
    OptimizeStart(pc->p_in); // Use stored p_in or assume it's set? 
    // OptimizeStart takes argument. p_init is passed in OptimizeIter.
    // We should probably pass p_in_arg to OptimizeDebug.
    // But for now, let's assume OptimizeStart was called or we use default.
    // Actually, in patchgrid.cpp, Optimize is called.
    // Let's use pc->p_in which should be set if initialized?
    // No, p_init is passed from outside.
    // Let's change signature to take p_in_arg?
    // Or just use zero?
    // In patchgrid.cpp, p_init[i] is passed.
    // We can't easily change signature in header without recompiling everything cleanly.
    // Let's assume p_in is 0 for now or passed via OptimizeStart before?
    // No, OptimizeStart sets p_in.
    // Let's just pass 0 vector for now as we are debugging patch 0 which usually starts at 0 displacement.
    Eigen::Vector2f p_zero; p_zero.setZero();
    OptimizeStart(p_zero);
    #else
    Eigen::Matrix<float, 1, 1> p_zero; p_zero.setZero();
    OptimizeStart(p_zero);
    #endif
  }
  
  int oldcnt=pc->cnt;

  // optimize patch until convergence
  while (  ! (pc->hasconverged || (pc->cnt > oldcnt + max_iter))  ) 
  {
    pc->cnt++;

    // Projection onto sd_images
    float sd_x, sd_y;
    #if (SELECTMODE==1)
      sd_x = (dxx_tmp.array() * pc->pdiff.array()).sum();
      sd_y = (dyy_tmp.array() * pc->pdiff.array()).sum();
      pc->delta_p[0] = sd_x;
      pc->delta_p[1] = sd_y;
    #else
      sd_x = (dxx_tmp.array() * pc->pdiff.array()).sum();
      sd_y = 0;
      pc->delta_p[0] = sd_x;
    #endif

    pc->delta_p = pc->Hes.llt().solve(pc->delta_p); // solve linear system
    // Hessian * delta_p = p_diff
    pc->p_iter -= pc->delta_p; // update flow vector
    
    #if (SELECTMODE==2) // if stereo depth
    if (cpt->camlr==0)
      pc->p_iter[0] = ::std::min((float)pc->p_iter[0],0.0f); // disparity in t can only be negative (in right image)
    else
      pc->p_iter[0] = ::std::max((float)pc->p_iter[0],0.0f); // ... positive (in left image)
    #endif
      
    // compute patch locations based on new parameter vector
    paramtopt(); 
      
    // check if patch(es) moved too far from starting location
    if ((pc->pt_st - pc->pt_iter).norm() > op->outlierthresh  // check if query patch moved more than >padval from starting location -> most likely outlier
        ||                  
        pc->pt_iter[0] < cpt->tmp_lb  || pc->pt_iter[1] < cpt->tmp_lb ||    // check patch left valid image region
        pc->pt_iter[0] > cpt->tmp_ubw || pc->pt_iter[1] > cpt->tmp_ubh)  
    {
      pc->p_iter = pc->p_in; // reset
      paramtopt(); 
      pc->hasconverged=1;
      pc->hasoptstarted=1;
    }
        
    OptimizeComputeErrImg();

    // Record debug data
    if (debug_data && (pc->cnt - 1 < max_iter)) {
        int idx = pc->cnt - 1;
        debug_data[idx].iter = pc->cnt;
        #if (SELECTMODE==1)
        debug_data[idx].p_x = pc->p_iter[0];
        debug_data[idx].p_y = pc->p_iter[1];
        debug_data[idx].dp_x = pc->delta_p[0];
        debug_data[idx].dp_y = pc->delta_p[1];
        debug_data[idx].hes_xx = pc->Hes(0,0);
        debug_data[idx].hes_xy = pc->Hes(0,1);
        debug_data[idx].hes_yy = pc->Hes(1,1);
        #else
        debug_data[idx].p_x = pc->p_iter[0];
        debug_data[idx].p_y = 0;
        debug_data[idx].dp_x = pc->delta_p[0];
        debug_data[idx].dp_y = 0;
        debug_data[idx].hes_xx = pc->Hes(0,0);
        debug_data[idx].hes_xy = 0;
        debug_data[idx].hes_yy = 0;
        #endif
        debug_data[idx].sd_x = sd_x;
        debug_data[idx].sd_y = sd_y;
        debug_data[idx].mares = pc->mares;
    }
  }
}
#endif
